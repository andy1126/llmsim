import glob
import os

import pytest

from src.config.model_config import HybridAttnConfig, ModelConfig
from src.layers.attn import LinearAttn, MHAAttn, MLAAttn
from src.models.base import ModelRegistry
from src.server_args import ServerArgs


def get_config_files():
    # 获取 hf_config 目录下所有的 json 文件
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return glob.glob(os.path.join(base_path, "hf_config/*.json"))
    # return glob.glob(
    #     os.path.join(base_path, "hf_config/qwen3-next-80B-A3B_config.json")
    # )


@pytest.mark.parametrize("config_path", get_config_files())
def test_weights_size_calculation(config_path):
    print(f"\nTesting config: {os.path.basename(config_path)}")

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

        # 遍历所有层进行详细验证
        for layer_idx in range(config.num_hidden_layers):
            layer = model.decode_blocks.blocks[layer_idx]
            attn = layer.attn

            # 4. 计算权重大小
            weight_size = attn.weights_size()
            total_weight_size += weight_size

            # 统计层类型
            t_name = type(attn).__name__
            layer_stats[t_name] = layer_stats.get(t_name, 0) + 1

            # 5. 验证
            assert (
                weight_size >= 0
            ), f"Weight size should be non-negative for {config_path} layer {layer_idx}"

            # 验证混合架构切换逻辑
            if isinstance(config.attn_config, HybridAttnConfig):
                if layer_idx % config.attn_config.full_attention_interval == 0:
                    assert isinstance(
                        attn, (MHAAttn, MLAAttn)
                    ), f"Layer {layer_idx} should be Full Attn"
                    # print(
                    #     f"Layer {layer_idx} is Full Attn, weight_size: {weight_size/ (1024**2):.2f}MB"
                    # )
                else:
                    assert isinstance(
                        attn, LinearAttn
                    ), f"Layer {layer_idx} should be LinearAttn"
                    # print(
                    #     f"Layer {layer_idx} is Linear Attn, weight_size: {weight_size/ (1024**2):.2f}MB"
                    # )

        # 打印统计信息
        stats_str = ", ".join([f"{k}: {v}" for k, v in layer_stats.items()])
        print(f"  FP8={use_fp8}: Layers distribution -> {stats_str}")
        print(
            f"    Total weight_size = {total_weight_size / (1024**3):.4f} GB ({total_weight_size / (1024**2):.2f} MB)"
        )


if __name__ == "__main__":
    # 如果直接运行脚本
    files = get_config_files()
    for f in files:
        try:
            test_weights_size_calculation(f)
        except Exception as e:
            print(f"Error testing {f}: {e}")
