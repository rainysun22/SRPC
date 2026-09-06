"""SR-PC（自省式预测编码）Phase-0 最小原型。

实现 docs/SRPC_DESIGN.md 第 7 节：单一自由能预测规则 + 自省环，
在交互中自发形成-修正-组合，全程局部在线、免反向传播。
"""
from .config import (AcceptanceConfig, FieldConfig, ModelConfig, SlotConfig,
                     Track1Config, Track2Config)
from .env import SlotWorld, SourceFieldWorld
from .model import SRPCModel, FlatPCModel, LookupModel
from .runner import (run_field, run_slot, run_track1, run_track2,
                     evaluate_acceptance)

__version__ = "0.1.0"

__all__ = [
    "ModelConfig", "FieldConfig", "SlotConfig", "Track1Config", "Track2Config",
    "AcceptanceConfig", "SourceFieldWorld", "SlotWorld",
    "SRPCModel", "FlatPCModel", "LookupModel",
    "run_field", "run_slot", "run_track1", "run_track2", "evaluate_acceptance",
]
