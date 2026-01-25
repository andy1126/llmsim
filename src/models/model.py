from src.models.base import BaseModel, ModelRegistry


@ModelRegistry.register("deepseek_v3")
@ModelRegistry.register("deepseek_v32")
@ModelRegistry.register("deepseek_v2")
class DeepSeekModel(BaseModel):
    # 如果 DeepSeek 有特殊的层级结构或计算公式，可以在这里重写方法
    pass


@ModelRegistry.register("qwen3")
class QwenModel(BaseModel):
    pass
