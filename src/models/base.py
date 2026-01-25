from typing import Dict, Type

from src.config.model_config import ModelConfig
from src.hardware import chip
from src.layers.decode_block import DecoderBlock, DecoderBlocks
from src.server_args import ServerArgs


class ModelRegistry:
    _registry: Dict[str, Type["BaseModel"]] = {}

    @classmethod
    def register(cls, model_type: str):
        def wrapper(model_cls: Type["BaseModel"]):
            cls._registry[model_type] = model_cls
            return model_cls

        return wrapper

    @classmethod
    def create(cls, serverArgs: ServerArgs, config: ModelConfig) -> "BaseModel":
        model_type = getattr(config, "model_type", "default")
        model_cls = cls._registry.get(model_type, BaseModel)
        return model_cls(serverArgs, config)


class BaseModel:
    def __init__(self, serverArgs: ServerArgs, config: ModelConfig):
        self.serverArgs = serverArgs
        self.config = config
        # get the chip message from the chip module
        self.chip = chip.chip_map[serverArgs.device_type]
        self.decode_blocks = DecoderBlocks(serverArgs, config)

    def get_decode_block(self, layer_idx: int) -> DecoderBlock:
        return self.decode_blocks.blocks[layer_idx]

    def total_weights_size(self):

        return self.decode_blocks.weights_bytes()
