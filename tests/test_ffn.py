import glob
import os

import pytest

from src.config.model_config import ModelConfig
from src.layers.ffn import DenseMLP, MoE
from src.models.base import ModelRegistry
from src.server_args import ServerArgs


def get_config_files():
    # 获取 hf_config 目录下所有的 json 文件
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # return glob.glob(os.path.join(base_path, "hf_config/*.json"))
    return glob.glob(
        os.path.join(base_path, "hf_config/qwen3-next-80B-A3B_config.json")
    )


@pytest.mark.parametrize("config_path", get_config_files())
def test_ffn_weights_size_calculation(config_path):
    print(f"\nTesting FFN for config: {os.path.basename(config_path)}")

    # 1. 加载配置
    config = ModelConfig.from_config_path(config_path)

    # 2. 准备 ServerArgs (分别测试 FP16 和 FP8)
    for use_fp8 in [False, True]:
        server_args = ServerArgs(config_path=config_path, use_fp8_gemm=use_fp8)

        total_weight_size = 0
        layer_stats = {}

        # 使用工厂模式创建模型实例
        model = ModelRegistry.create(server_args, config)
        print(f"  Model instance type: {type(model).__name__}")

        # 遍历所有层验证 FFN
        for layer_idx in range(config.num_hidden_layers):
            layer = model.decode_blocks.blocks[layer_idx]
            ffn = layer.ffn

            # 计算权重大小
            weight_size = ffn.weights_size()
            total_weight_size += weight_size

            # 统计层类型
            t_name = type(ffn).__name__
            layer_stats[t_name] = layer_stats.get(t_name, 0) + 1

            # 验证权重非负
            assert (
                weight_size >= 0
            ), f"FFN weight size should be non-negative for {config_path} layer {layer_idx}"

            # 验证 DeepSeek 等模型的 first_k_dense_replace 逻辑
            if config.first_k_dense_replace > 0:
                if layer_idx < config.first_k_dense_replace:
                    print(
                        f"  Layer {layer_idx} is DenseMLP, size: {weight_size/ (1024**2):.2f}MB"
                    )
                    assert isinstance(
                        ffn, DenseMLP
                    ), f"Layer {layer_idx} should be DenseMLP (first_k_dense_replace={config.first_k_dense_replace})"
                elif config.moe_config and config.moe_config.num_routed_experts > 1:
                    print(
                        f"  Layer {layer_idx} is MoE, size: {weight_size/ (1024**2):.2f}MB"
                    )
                    # 注意：如果 layer_idx >= first_k_dense_replace 且配置了 MoE，应该是 MoE
                    assert isinstance(ffn, MoE), f"Layer {layer_idx} should be MoE"
            elif config.moe_config and config.moe_config.num_routed_experts > 1:
                print(
                    f"  Layer {layer_idx} is MoE, size: {weight_size/ (1024**2):.2f}MB"
                )

        # 打印统计信息
        stats_str = ", ".join([f"{k}: {v}" for k, v in layer_stats.items()])
        print(f"  FP8={use_fp8}: FFN Layers distribution -> {stats_str}")
        print(
            f"    Total FFN weight_size = {total_weight_size / (1024**3):.4f} GB ({total_weight_size / (1024**2):.2f} MB)"
        )


if __name__ == "__main__":
    # 如果直接运行脚本
    files = get_config_files()
    for f in files:
        try:
            test_ffn_weights_size_calculation(f)
        except Exception as e:
            print(f"Error testing {f}: {e}")
