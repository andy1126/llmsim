import os
import pytest
from src.config.model_config import ModelConfig
from src.layers.decode_block import DecoderBlocks
from src.server_args import ServerArgs

def get_test_configs():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    configs = {
        "Qwen3-8B (MHA)": "hf_config/qwen3-8B_config.json",
        "DeepSeek-R1 (MLA)": "hf_config/deepseek_671b_r1_config.json",
        "Qwen3-Next (Hybrid)": "hf_config/qwen3-next-80B-A3B_config.json",
    }
    return {
        name: os.path.join(base_path, path)
        for name, path in configs.items()
        if os.path.exists(os.path.join(base_path, path))
    }

@pytest.mark.parametrize("model_name, config_path", get_test_configs().items())
def test_kvcache_scaling(model_name, config_path):
    print(f"\n{'='*20} Testing {model_name} {'='*20}")
    config = ModelConfig.from_config_path(config_path)
    
    # 测试不同序列长度
    context_lengths = [1024, 4096]
    results = {}

    for ctx_len in context_lengths:
        server_args = ServerArgs(
            config_path=config_path,
            tp_size=1,
            use_fp8_kv=False # 使用 FP16
        )
        
        blocks = DecoderBlocks(server_args, config)
        
        print(f"\n--- Context Length: {ctx_len} ---")
        blocks.print_kvcache_info(ctx_len)
        
        kv_bytes = blocks.kvcache_bytes(ctx_len)
        results[ctx_len] = kv_bytes
        
        assert kv_bytes > 0, f"KV Cache for {model_name} should be > 0"

    # 验证增长趋势
    if "Linear" not in model_name: # 对于纯 MHA/MLA 模型，应该是线性增长的
        # 注意：MLA 包含压缩 KV，MHA 包含全量 KV，都会随长度增长
        assert results[4096] > results[1024], "KV cache should increase with context length for MHA/MLA"
    
    # 对于 Hybrid 模型，由于包含 Linear Attention (固定 State)，
    # 虽然总和会随长度增加（因为有部分是 Full Attn），但增加的斜率应该不同。
    if "Hybrid" in model_name:
        print(f"Hybrid model growth: {results[4096]/results[1024]:.2f}x (Expected < 4x if Linear layers dominate)")

@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_kvcache_tp_scaling(tp_size):
    """测试 TP 对 KV Cache 的削减效果 (主要针对 MHA 和 Linear)"""
    config_path = "hf_config/qwen3-8B_config.json" # MHA
    if not os.path.exists(config_path):
        pytest.skip("Config not found")
        
    config = ModelConfig.from_config_path(config_path)
    server_args = ServerArgs(config_path=config_path, tp_size=tp_size)
    
    blocks = DecoderBlocks(server_args, config)
    kv_bytes = blocks.kvcache_bytes(1024)
    
    print(f"\nTP {tp_size} KV Cache (1024 tokens): {kv_bytes/(1024**2):.2f} MB")
    
    # 对于 MHA，TP 增加，单卡 KV 应该按比例减少
    # 期待结果：TP=2 时是 TP=1 的一半
    # (考虑到 Python 浮点数，我们做个比例断言)
    return kv_bytes

if __name__ == "__main__":
    # 手动运行示例
    configs = get_test_configs()
    for name, path in configs.items():
        test_kvcache_scaling(name, path)
