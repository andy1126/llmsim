from src.config.model_config import ModelConfig
from src.models.base import BaseModel, ModelRegistry
from src.server_args import ServerArgs


@ModelRegistry.register("qwen3_next")
class Qwen3Next(BaseModel):
    def __init__(self, serverArgs: ServerArgs, config: ModelConfig):
        super().__init__(serverArgs, config)
