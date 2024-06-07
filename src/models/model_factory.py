from src.configs import ModelConfig
from .model import BaseModel
from .hf_model import HFModel


def get_model(config: ModelConfig) -> BaseModel:
    if config.provider == "hf":
        return HFModel(config)
    else:
        raise NotImplementedError
    