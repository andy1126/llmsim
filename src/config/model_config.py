import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


@dataclass
class MHAConfig:
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int


@dataclass
class MLAConfig:
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    qk_head_dim: int
    v_head_dim: int


@dataclass
class LinearAttnConfig:
    conv_kernel_dim: int
    key_head_dim: int
    num_key_heads: int
    value_head_dim: int
    num_value_heads: int


@dataclass
class HybridAttnConfig:
    full_attn_config: Union[MHAConfig, MLAConfig]
    linear_attn_config: LinearAttnConfig
    full_attention_interval: int


@dataclass
class MoEConfig:
    num_routed_experts: int
    num_experts_per_tok: int
    intermediate_size: int
    num_shared_experts: int = 0


@dataclass
class ModelConfig:
    hidden_size: int
    num_hidden_layers: int
    # 模型类型，用于路由到不同的 Model 类
    model_type: str
    # 核心拆分：使用 Union 表达多种可能的注意力配置
    attn_config: Union[MHAConfig, MLAConfig, LinearAttnConfig, HybridAttnConfig]
    # FFN/MoE 拆分
    moe_config: Optional[MoEConfig] = None
    # 混合层信息
    is_hybrid: bool = False
    num_full_attn_layers: int = 0
    num_linear_attn_layers: int = 0
    # 其他全局信息
    first_k_dense_replace: int = 0
    # 保存原始 JSON 配置，方便访问未显式定义的字段
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config_path(cls, config_path):
        with open(config_path, "r") as f:
            d = json.load(f)

        hidden_size = d["hidden_size"]
        num_hidden_layers = d["num_hidden_layers"]

        # 1. 解析 Attention 配置
        is_hybrid = d.get("full_attention_interval") is not None

        # 基础 MHA 配置
        mha_config = None
        if "num_attention_heads" in d:
            head_dim = d.get("head_dim", hidden_size // d["num_attention_heads"])
            mha_config = MHAConfig(
                num_attention_heads=d["num_attention_heads"],
                num_key_value_heads=d["num_key_value_heads"],
                head_dim=head_dim,
            )

        # 基础 MLA 配置
        mla_config = None
        if "kv_lora_rank" in d:
            mla_config = MLAConfig(
                num_attention_heads=d["num_attention_heads"],
                q_lora_rank=d["q_lora_rank"],
                kv_lora_rank=d["kv_lora_rank"],
                qk_nope_head_dim=d["qk_nope_head_dim"],
                qk_rope_head_dim=d["qk_rope_head_dim"],
                qk_head_dim=d["qk_nope_head_dim"] + d["qk_rope_head_dim"],
                v_head_dim=d["v_head_dim"],
            )

        # 基础 Linear 配置
        linear_config = None
        if "linear_key_head_dim" in d:
            linear_config = LinearAttnConfig(
                conv_kernel_dim=d["linear_conv_kernel_dim"],
                key_head_dim=d["linear_key_head_dim"],
                num_key_heads=d["linear_num_key_heads"],
                value_head_dim=d["linear_value_head_dim"],
                num_value_heads=d["linear_num_value_heads"],
            )

        # 组合逻辑
        if is_hybrid and (mla_config or mha_config) and linear_config:
            full_cfg = mla_config or mha_config
            assert full_cfg is not None
            attn_config = HybridAttnConfig(
                full_attn_config=full_cfg,
                linear_attn_config=linear_config,
                full_attention_interval=d["full_attention_interval"],
            )
        elif mla_config:
            attn_config = mla_config
        elif linear_config:
            attn_config = linear_config
        else:
            assert mha_config is not None
            attn_config = mha_config

        # 2. 解析 MoE 配置
        moe_config = None
        if any(
            k in d for k in ["num_routed_experts", "num_experts", "n_routed_experts"]
        ):
            num_routed = d.get(
                "num_routed_experts", d.get("num_experts", d.get("n_routed_experts"))
            )
            moe_inter_size = d.get("moe_intermediate_size", d.get("intermediate_size"))
            num_shared = d.get("num_shared_experts", d.get("n_shared_experts", 0))

            moe_config = MoEConfig(
                num_routed_experts=num_routed,
                num_experts_per_tok=d["num_experts_per_tok"],
                intermediate_size=moe_inter_size,
                num_shared_experts=num_shared,
            )
        else:
            # Dense FFN 也可以复用这个结构，或者单独处理
            moe_config = MoEConfig(
                num_routed_experts=1,
                num_experts_per_tok=1,
                intermediate_size=d["intermediate_size"],
            )

        # 3. 解析混合层信息
        is_hybrid = d.get("full_attention_interval") is not None
        num_full = 0
        num_linear = 0
        if is_hybrid:
            num_full = num_hidden_layers // d["full_attention_interval"]
            num_linear = num_hidden_layers - num_full

        return cls(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            model_type=d.get("model_type", "default"),
            attn_config=attn_config,
            moe_config=moe_config,
            is_hybrid=is_hybrid,
            num_full_attn_layers=num_full,
            num_linear_attn_layers=num_linear,
            first_k_dense_replace=d.get("first_k_dense_replace", 0),
            raw_config=d,
        )
