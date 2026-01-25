import glob
import os

import pytest

from src.config.model_config import ModelConfig
from src.layers.decode_block import DecoderBlocks
from src.server_args import ServerArgs


def get_config_files():
    # 获取 hf_config 目录下所有的 json 文件
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return glob.glob(os.path.join(base_path, "hf_config/*.json"))
    # return glob.glob(os.path.join(base_path, "hf_config/deepseek_671b_r1_config.json"))


@pytest.mark.parametrize("config_path", get_config_files())
def test_print_decode_weights_info(config_path):
    print(
        f"\nTesting print_decode_block_weights_info for config: {os.path.basename(config_path)}"
    )

    # 1. 加载配置
    config = ModelConfig.from_config_path(config_path)

    # 2. 准备 ServerArgs
    server_args = ServerArgs(
        config_path=config_path, use_fp8_gemm=True, world_size=8, ep_size=8
    )

    # 3. 创建 DecoderBlocks
    decoder_blocks = DecoderBlocks(server_args, config)

    # 4. 调用打印方法
    # 注意：我们主要验证它是否能正常运行不报错
    decoder_blocks.print_decode_block_weights_info()

    # 5. 基本断言
    assert len(decoder_blocks.blocks) == config.num_hidden_layers
    assert decoder_blocks.weights_bytes() >= 0
    assert decoder_blocks.total_attn_weights() >= 0
    assert decoder_blocks.total_ffn_weights() >= 0


if __name__ == "__main__":
    # 如果直接运行脚本
    files = get_config_files()
    for f in files:
        try:
            test_print_decode_weights_info(f)
        except Exception as e:
            print(f"Error testing {f}: {e}")
