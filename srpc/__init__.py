"""SR-PC（自省式预测编码）演化智能系统。

Phase-0（阶段 A）：最小原型 —— 单一自由能预测规则 + 自省环，形成-修正-组合。
Phase-B（阶段 B）：规模化与组合 —— 深层预测编码（DeepSRPC）、多时间尺度记忆
（PrototypeMemory）、ARC-lite 组合基准、顺序学习免遗忘验收。

全程 NumPy 局部规则、免反向传播（docs/SRPC_DESIGN.md）。
"""
from .config import (AcceptanceBConfig, AcceptanceConfig, ArcConfig, CLConfig,
                     DeepConfig, FieldConfig, MemoryConfig, ModelConfig,
                     SlotConfig, Track1Config, Track2Config)
from .env import SlotWorld, SourceFieldWorld
from .model import SRPCModel, FlatPCModel, LookupModel
from .runner import (run_field, run_slot, run_track1, run_track2,
                     evaluate_acceptance)
from .memory import PrototypeMemory
from .arc import ArcLite
from .deepmodel import DeepSRPC
from .runner_b import (eval_task, eval_combination, run_sequential,
                       forget_stats, evaluate_acceptance_b)

__version__ = "0.2.0"

__all__ = [
    "ModelConfig", "FieldConfig", "SlotConfig", "Track1Config", "Track2Config",
    "AcceptanceConfig", "DeepConfig", "MemoryConfig", "ArcConfig", "CLConfig",
    "AcceptanceBConfig", "SourceFieldWorld", "SlotWorld",
    "SRPCModel", "FlatPCModel", "LookupModel",
    "run_field", "run_slot", "run_track1", "run_track2", "evaluate_acceptance",
    "PrototypeMemory", "ArcLite", "DeepSRPC",
    "eval_task", "eval_combination", "run_sequential", "forget_stats",
    "evaluate_acceptance_b",
]
