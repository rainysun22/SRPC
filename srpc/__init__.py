"""SR-PC（自省式预测编码）演化智能系统。

Phase-0（阶段 A）：最小原型 —— 单一自由能预测规则 + 自省环，形成-修正-组合。
Phase-B（阶段 B）：规模化与组合 —— 深层预测编码（DeepSRPC）、多时间尺度记忆
（PrototypeMemory）、ARC-lite 组合基准、顺序学习免遗忘验收。
Phase-C（阶段 C，软件版内在化）：出生即结构稀疏（扇入受限分块权重 + k-WTA）
+ 三口径有效 MAC 能耗记账 + 大模型标尺对比 + int8 部署就绪检查
（硬件部署本体 deferred 至有专用硬件）。

全程 NumPy 局部规则、免反向传播（docs/SRPC_DESIGN.md）。
"""
from .config import (AcceptanceBConfig, AcceptanceCConfig, AcceptanceConfig,
                     ArcConfig, CLConfig, CreditConfig, DeepConfig, FieldConfig,
                     MemoryConfig, ModelConfig, PhaseCConfig, SlotConfig,
                     Track1Config, Track2Config)
from .env import SlotWorld, SourceFieldWorld
from .model import SRPCModel, FlatPCModel, LookupModel
from .runner import (run_field, run_slot, run_track1, run_track2,
                     evaluate_acceptance)
from .credit import run_credit_screen
from .memory import PrototypeMemory
from .arc import ArcLite
from .deepmodel import DeepSRPC
from .energy import EnergyLedger, llm_task_macs
from .runner_b import (eval_task, eval_combination, run_sequential,
                       forget_stats, evaluate_acceptance_b)
from .runner_c import (make_sparse_cfg, make_dense_cfg, run_phase_c,
                       measure_inference, structural_stats, int8_check)

__version__ = "0.3.0"

__all__ = [
    "ModelConfig", "FieldConfig", "SlotConfig", "Track1Config", "Track2Config",
    "AcceptanceConfig", "CreditConfig", "DeepConfig", "MemoryConfig",
    "ArcConfig", "CLConfig",
    "AcceptanceBConfig", "PhaseCConfig", "AcceptanceCConfig",
    "SourceFieldWorld", "SlotWorld",
    "SRPCModel", "FlatPCModel", "LookupModel",
    "run_field", "run_slot", "run_track1", "run_track2", "evaluate_acceptance",
    "run_credit_screen",
    "PrototypeMemory", "ArcLite", "DeepSRPC",
    "EnergyLedger", "llm_task_macs",
    "eval_task", "eval_combination", "run_sequential", "forget_stats",
    "evaluate_acceptance_b",
    "make_sparse_cfg", "make_dense_cfg", "run_phase_c", "measure_inference",
    "structural_stats", "int8_check",
]
