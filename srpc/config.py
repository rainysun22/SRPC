"""SR-PC Phase-0 全部超参数（对齐 docs/SRPC_DESIGN.md 第 7 节）。

所有实验的唯一参数来源，保证可复现。
"""
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """SR-PC 核心模型（7.2 模块清单 / 7.3 更新规则）。

    层级（自上而下生成、自下而上只传误差）：
        x_self --Ws2--> x2 --W21--> x1 --W10--> s_hat ~= s
        自省环：Wdyn 预测 x_self(t+1) | x_self(t), a(t)  -->  e_self
    """

    d_obs: int = 32            # 感觉维度 D（低维环境感知）
    n_l1: int = 24             # L1：感觉通路第一层（感受野/部件层）
    n_l2: int = 16             # L2：感觉通路第二层（概念/组合层）
    n_self: int = 16           # x_self：自我模型层（不变量 4：从一开始就在）
    x_max: float = 5.0         # 状态上界（非负稀疏表征）
    inner_iters: int = 3       # 每步局部推断迭代次数
    # --- 7.3 规则 1：局部推断（稀疏/事件驱动） ---
    alpha: float = 0.15        # 系数 a：来自上层的预测误差（拉动 x_l 向上层预测靠拢）
    beta: float = 0.25         # 系数 b：来自下层的误差信号（自下而上只传误差）
    theta_event: float = 0.01  # 事件阈值：|更新|>theta 的节点才更新（不变量 3：能量内生·稀疏）
    # --- 7.3 规则 2：局部 Hebbian 学习（免反传） ---
    eta_w: float = 0.06        # 生成权重（W10/W21/Ws2）学习率
    theta_syn: float = 1e-2    # 突触前活跃门限：仅活跃节点的突触列参与更新（稀疏学习）
    # --- 7.3 规则 3：自省环 ---
    eta_dyn: float = 0.03      # 自我预测器 Wdyn 的局部 LMS 学习率
    dyn_decay: float = 1e-3    # Wdyn 权重衰减（稳定性）
    kappa_boost: float = 1.5   # 精度调制增益：eta_t = eta*(1+kappa*(surprise-1))^+，上限 boost_max
    boost_max: float = 6.0
    ema_self_rate: float = 0.005  # ||e_self|| 的 EMA 基线（慢适应：扰动后 boost 可覆盖重学期）
    u_action_rate: float = 0.10   # 每动作"预期自省误差"EMA 更新率（7.2 模块 5）
    u_regress: float = 0.005      # 久未执行动作的 U 向均值回归率（不确定性回升，保持探索）
    curiosity: float = 3.0        # 主动推理的认识价值项系数：少试的动作按 1/sqrt(n) 折价探索
    # --- 7.2 模块 4：全局工作空间（MVP：高置信收敛信念子集） ---
    ws_k: int = 4              # 广播的 top-k 信念
    # --- 结构性稀疏（不变量 3：阶段 A 起就是核心算子的数学形式，非训练后裁剪） ---
    # 阶段 A 网络小（24/16/16 维），密度 0.75 维持自组织稳定性（3 seeds 自省增益全过）；
    # 阶段 B 网络大（160/128/112），DeepConfig 用 0.25 已验。
    fan_in_frac: float = 0.75     # 生成权重每列扇入占比（层1=连续感受野窗口，内部层=随机扇入）
    fan_in_dyn_frac: float = 0.75 # 自省动力学 Wdyn 每列扇入占比
    kwta_frac: float = 0.5        # k-WTA：每层保留 top-k 激活占比（结构性稀疏激活）


@dataclass
class FieldConfig:
    """Track-1 环境：SourceFieldWorld（2-D 场地 + 信息源，交互式导航）。"""

    n_sources: int = 5
    source_pos: tuple = ((0.20, 0.20), (0.80, 0.20), (0.20, 0.80), (0.80, 0.80), (0.50, 0.50))
    kernel_sigma: float = 0.13   # 邻近核宽度（决定单源/组合感受范围）
    move_step: float = 0.12
    move_noise: float = 0.02
    obs_noise: float = 0.03
    pattern_active: int = 7      # 每个源模式的活跃维数（相邻源轻度重叠）
    pattern_stride: int = 5      # 活跃维块间距（< active 表示相邻源共享若干维）
    # 修正测试扰动：全局感觉重映射（感知维度随机置换，无论智能体在哪都被迫重学信念）
    perturb_kind: str = "remap"


@dataclass
class SlotConfig:
    """Track-2 环境：SlotWorld（合成组合流，验证"片段重组出新概念"，7.4 组合）。"""

    n_features: int = 6
    # 15 个特征对中留 4 个作保留组合（训练不可见），每个特征仍在训练对中出现
    train_pairs: tuple = ((0, 1), (0, 2), (0, 3), (0, 5), (1, 2), (1, 3), (1, 4),
                          (2, 3), (2, 5), (3, 4), (4, 5))
    novel_pairs: tuple = ((0, 4), (1, 5), (2, 4), (3, 5))
    single_prob: float = 0.25    # 训练流中单特征上下文占比（提供"纯净部件"片段）
    ctx_min: int = 20            # 一个上下文持续的最少步数（跨时段结构）
    ctx_max: int = 60
    obs_noise: float = 0.03
    pattern_active: int = 7
    pattern_stride: int = 5


@dataclass
class Track1Config:
    """Track-1 实验协议（形成/修正/自省 A/B）。"""

    steps: int = 12000
    perturb_step: int = 6000
    eps_start: float = 0.40      # 主动推理的探索率（探索/利用均衡）
    eps_end: float = 0.20
    eps_decay_steps: int = 4000
    ema_alpha: float = 0.01      # 误差曲线平滑
    recovery_base_win: int = 150
    recovery_mult: float = 1.30  # 恢复阈值 = 扰动前基线 * mult
    recovery_sustain: int = 60
    relearn_lo: int = 300        # 扰动后"重学窗口"起点（跳过 EMA 滞后）
    relearn_hi: int = 2500       # 重学窗口终点


@dataclass
class Track2Config:
    """Track-2 实验协议（组合泛化 + 基线对照 + 自省增益）。"""

    train_steps: int = 6000
    fewshot_steps: int = 500     # 保留组合的少样本在线适应阶段
    probe_samples: int = 30      # 冻结评估时每个上下文的采样数


@dataclass
class CreditConfig:
    """§8.5 信用分配早筛（承重墙）任务与验收阈值（阶段 A 通过条件第 2 项）。

    任务：延迟 XOR —— 目标 y_t = XOR(bit0(x_{t-Δ}), bit1(x_{t-Δ}))，
    输入为最近 Δ+1 步窗口拼接，远端块（t-Δ）为唯一任务相关块、其余块
    为随机干扰（每步 2 维 = 双峰 bit 对；干扰来自其它时间块）。XOR 目标
    与任何单一输入特征零边际相关（纯相关 Hebbian 必然失败），且线性不可分
    （输出必须为 one-hot 双输出单元）。
    网络与 Phase-0 同一套 7.3 局部规则（出生即稀疏：掩码 + k-WTA）；
    对照 = 纯相关 Hebbian（§2.4：朴素 Hebbian ≠ 误差驱动 PCN）。
    判据：误差驱动显著优于纯相关（误差机制承载信用分配），且误差能量
    回传驱动远端权重（远端块权重变化占比 > 机会水平 1/(Δ+1)）。

    **种子间方差修复（v2，3 seeds 全过）**：
    - bit 编码 0/1（原 0.2/0.8）：低输入强度位型 (0,0) 的远端感知弱是
      seed1 掉队根因（dbg27：位型准确率 0.425 < 随机）；0/1 双峰距离最大，
      低强度位型同样产生可区分表征，XOR 语义不变；
    - alpha=1.5（更强顶层类拉动）：类质心坍缩缓解（x2cos 0.886→分离）；
    - err 臂深迭代收敛（settle_iters=32, eta_inf=0.09）；
    - hebb 臂浅迭代（hebb_settle_iters=8）+ 自由推断训练（hebb_free）：
      纯相关是瞬时联想、无需深迭代；钳制 yoh 会泄漏类信息抬高基线，
      自由推断（仅输入驱动表征）使基线贴近随机（~0.51），gap 判据成立。
    """

    # 任务几何
    d_feat: int = 2                # 每步特征维（前两维为双峰 bit；其余块为随机干扰）
    delay: int = 4                 # 长程延迟 Δ（目标取决于 Δ 步前的输入）
    bit_lo: float = 0.0            # bit 低电平编码（0/1 双峰，原 0.2/0.8 的方差修复）
    bit_hi: float = 1.0
    # 网络（x0 -> x1 -> x2 -> one-hot 输出，与 Phase-0 感官通路同构）
    h1: int = 32                   # L1 隐层维度（时间分块感受野；128/64→32/16 降谱半径）
    h2: int = 16                   # L2 隐层维度（随机扇入，联合特征层）
    x_max: float = 5.0
    settle_iters: int = 32         # 局部推断收敛迭代（误差回传的深度；err 臂）
    hebb_settle_iters: int = 8     # hebb 臂浅迭代：纯相关瞬时联想，深迭代反抬基线
    eta_inf: float = 0.09          # 推断阻尼步长（固定；谱半径 ~8 下的收敛步长）
    # 7.3 规则 1 系数：顶层类拉动 α 必须 >> 底层重建 β（XOR 类条件均值相同，
    # 重建误差无法分位，类分离只能由 top-down 原型拉动提供）
    alpha: float = 1.5
    beta: float = 1.0
    theta_event: float = 0.01
    eta_out: float = 0.1           # 自由输出模式下 x3 的推断步长
    # 7.3 规则 2
    eta_w: float = 0.05            # 权重学习率（信用分配臂调优值）
    theta_syn: float = 1e-2
    # 不变量 3：出生即结构稀疏（阶段 A 密度 0.75 维持自组织稳定性）
    fan_in_frac: float = 0.75
    kwta_frac: float = 0.5
    kwta_on: bool = True
    hebb_free: bool = True         # hebb 臂训练用自由推断（无 yoh 钳制，消除类泄漏）
    predict_mode: str = "compare"  # "compare" 钳制-比较 / "free" 自由推断+读出头
    energy_mode: str = "full"      # 分类比较口径："full"=e0+e1+e2 / "class"=e1+e2（排除类无关重建噪声）
    # 协议
    train_steps: int = 10000
    eval_steps: int = 1500        # 冻结评估样本数（与 verify_final 验证口径一致；500 下种子间噪声大）
    # 验收阈值（§8.5：需量化阈值；承重墙 = 承重墙，早筛不过立即回头）
    acc_pcn_min: float = 0.80      # 误差驱动在长程延迟上准确率 >= 80%（机会 50%）
    acc_hebb_max: float = 0.68     # 纯相关必须 <= 68%（否则判据不具区分力）
    acc_gap_min: float = 0.30      # 误差驱动 - 纯相关 >= 30 个百分点
    distal_share_min: float = 0.25 # 远端块权重变化占比 >= 25%（机会水平 20%）


@dataclass
class AcceptanceConfig:
    """7.5 验收标准的量化阈值（阶段 A 通过条件）。"""

    rf_unit_cos: float = 0.75        # 一个源/特征被"捕获"：存在感受野与其余弦 >= 该值
    rf_captured_frac: float = 0.80   # 至少 80% 的源/特征被捕获
    rf_improve_min: float = 0.15     # 感受野对齐相对初始化的最小提升
    nmi_p_max: float = 0.05          # 概念层-上下文 NMI 的置换检验 p 值上限
    slope_max: float = 0.0           # 预测误差 EMA 斜率必须 < 0（随交互下降）
    self_gain_min_frac: float = 0.10  # 自省环开启后扰动重学误差至少低 10%（A/B 主指标）


# ----------------------------------------------------------------------
# 阶段 B（规模化与组合）：DeepSRPC / 多时间尺度记忆 / ARC-lite / 顺序学习
# 对应 docs/SRPC_DESIGN.md 第 8 节里程碑 B
# ----------------------------------------------------------------------

@dataclass
class DeepConfig:
    """阶段 B 深层预测编码网络（规模化：Phase-0 的 3 层 -> 可配置 L 层）。

    dims[0] 为输入维（s），dims[-1] 为自我层维（x_self），内部层数 L = len(dims)-1。
    自上而下生成预测、自下而上只传误差；事件驱动稀疏（不变量 3）；
    自省环 + 记忆先验 + 变换条件经顶层接入（不变量 4 / 阶段 B 里程碑）。
    """

    dims: tuple = (256, 160, 128, 112, 128)  # 输入维(网格一热) + 内部层维 + 自我层维
    x_max: float = 5.0
    inner_iters: int = 3
    # 规则 1：局部推断（稀疏/事件驱动）
    # B 收尾 W1 实证：alpha=0.4 + kwta=0.8 提升 x1 输入保真度
    # （LS 冻结解 flip_h 色块格正确率 0.82 -> 0.90；kwta=0.5 时 top-50% 剪枝
    # 会丢掉色块细节，见 docs/SRPC_DESIGN.md §8 里程碑 B 收尾记录）。
    alpha: float = 0.40
    beta: float = 0.25
    # B 收尾 W1：推断步长。谱证据（probe_w1z 实测）：eig_max(W1^TW1) 出生
    # 47.97（连续感受野窗口重叠 + 非负元素 -> 列高相关）-> 训练后 3.00
    # （Hebbian + 列归一化去相关）。训练后线性谱已稳，>3 内迭代精度退化
    # （0.96 -> 0.69@8iters）源于非线性环：k-WTA 剪枝 + x_max 截断 + 顶层
    # 先验持续拉动的混沌传播（x2 触截断、x1 差异振荡放大）——架构债务 #2，
    # 由 E0 承接（见 results_phaseB/report_w1.md §3）。
    eta_inf: float = 1.0
    theta_event: float = 0.01
    # 规则 2：局部 Hebbian 学习（免反传）
    eta_w: float = 0.05
    theta_syn: float = 1e-2
    # 规则 3：自省环
    eta_dyn: float = 0.03
    dyn_decay: float = 1e-3
    kappa_boost: float = 1.5
    boost_max: float = 6.0
    ema_self_rate: float = 0.005
    # 阶段 B：多时间尺度记忆先验（顶层拉动，按任务分组免遗忘）
    gamma_mem: float = 0.35
    # 阶段 B：ARC 变换条件先验。
    # Uc 为固定分块正交码（非负、不相交支撑 -> 任务编码天然分离，见 set_condition）；
    # 条件直接拉动 x_self 更新（beta_cond 为拉动强度），变换知识由读出层 W_out 学习。
    beta_cond: float = 0.80
    # --- 读出头（翻译器）学习规则（B 收尾 W1 实证） ---
    # NLMS（旧默认）在病态 x1 特征上收敛极慢：微缩实验 3000 步后 flip_h 色块格
    # 正确率仅 ~0.40（e 停在 ~1.5-2），远低于 LS 冻结解上限 0.90 —— 病态条件数
    # 拖死梯度类步长。RLS（递归最小二乘，逐样本、逐输出维独立、免反传，标准
    # 自适应滤波）4000 步内逼近 LS 上限（cell 0.993 / 色块格 0.894，kwta0.8+alpha0.4）。
    # 读出头是 §7.7 机械外围翻译器（无自主动力学），RLS 的 P 矩阵为其内部状态，
    # 与核心的 Hebbian 局部规则不冲突（核心仍为误差驱动局部学习）。
    ro_alg: str = "rls"           # 读出头学习规则："rls"（递归最小二乘）/"nlms"（归一化 LMS，旧路径保留对照）
    ro_rls_lam: float = 0.999     # RLS 遗忘因子（1=无穷记忆；小则更快丢弃旧样本）
    eta_wout: float = 0.08      # 读出层 W_out 学习率（输入->输出映射；仅 nlms 使用）
    readout_gate: float = 1e-2  # 读出学习突触前活跃门限（仅 nlms 使用）
    ro_norm: str = "clip"     # 读出权重更新后归一化："full"=列单位 L2（与生成权重一致）/"clip"=列范数超 ro_norm_cap 时投影回该球面（保稳定性+允许大尺度）
    ro_norm_cap: float = 6.0  # clip 模式的列范数上限（LS 解列范数均值 ~1.5、最大 ~4.5；cap=6 留裕量防 runaway）
    # 读出头源模式（E0 记忆-读出耦合实验矩阵，v1.6）：
    #   "recon" = 核心重建 ŝ=W1@x1（默认主路径，符号空间，翻转/旋转/重着色为精确线性映射，
    #             读出头逼近置换矩阵；B 收尾 W1 实证：LS+随机扇入掩码 0.74 vs 稠密 0.99，
    #             随机掩码与置换结构冲突 -> 稠密头）
    #   "self"  = 顶层 x_L（E0-b1：记忆/条件先验直接拉动该层，耦合零传播延迟；
    #             风险 = 符号保真回退，旧配置 cell 0.92/grid 0.0，v2 阈值下重测）
    #   "dual"  = concat[ŝ, x_L]（E0-b2：保真走 ŝ 路径、记忆耦合走 x_L 路径各取所长）
    #   "x1"    = 原始 x1（旧对照路径，随机扇入掩码 + NLMS）
    ro_recon_mode: str = "recon"
    # --- 结构性稀疏（不变量 3：从阶段 A/B 起就是核心算子的数学形式，非训练后裁剪） ---
    # 出生即定型掩码 + k-WTA 激活；学习只更新已有突触（*= mask），结构由构造保证。
    fan_in_frac: float = 0.25     # 生成权重每列扇入占比（层1=连续感受野窗口，内部层=随机扇入）
    fan_in_ro_frac: float = 0.40  # 读出头每 self 维扇入占比
    fan_in_dyn_frac: float = 0.25 # 自省动力学 Wdyn 每列扇入占比
    kwta_frac: float = 0.80       # k-WTA：每层保留 top-k 激活占比（结构性稀疏激活）
    # B 收尾 W1：0.5 -> 0.8（LS 冻结解 flip_h 色块格 0.82 -> 0.90；kwta=0.5 剪掉色块细节）
    trace_energy: bool = False    # 有效 MAC 记账（阶段 C 部署度量；默认关，零开销）


@dataclass
class MemoryConfig:
    """阶段 B 多时间尺度原型记忆（免遗忘结构，不变量 2：能力=记忆·拼合）。

    fast 槽：工作记忆，学习率快，可快速覆盖（当前会话/任务）；
    slow 槽：长期记忆，按任务分组（n_slow_group 个/组），学习率慢，
    只在"稳定/重复"时小幅巩固；顺序学习新任务只写新任务的组，
    旧任务原型固化在旧组 -> 结构上免遗忘（新任务不破坏旧任务）。
    """

    d: int = 12
    n_fast: int = 12
    n_slow_group: int = 2       # 每任务组的长期记忆槽数
    rate_fast: float = 0.15
    rate_slow: float = 0.01
    conf_thresh: float = 0.30   # 自省误差小（stable>=thresh）才巩固


@dataclass
class ArcConfig:
    """阶段 B 组合基准：ARC-lite（8x8 网格变换任务 + 保留组合零样本）。

    训练变换作为顺序学习任务序列；novel_combos 为两个已知变换的组合
    （训练不可见），验证"既有片段重组出新概念"（阶段 B 里程碑）。
    """

    grid: int = 8
    n_colors: int = 4           # 0=空, 1..3 颜色
    n_blocks: int = 3           # 每输入随机放置的块数上限
    max_block: int = 2          # 块最大边长
    train_transforms: tuple = ("flip_h", "flip_v", "rot90", "recolor")
    novel_combos: tuple = (("flip_h", "rot90"), ("recolor", "flip_v"))


@dataclass
class CLConfig:
    """阶段 B 顺序学习（免遗忘）协议。"""

    steps_per_task: int = 800   # 每个变换任务的训练步数
    eval_samples: int = 24      # 冻结评估每任务采样数
    settle: int = 6             # 评估收敛步（跳过瞬态）


@dataclass
class AcceptanceBConfig:
    """阶段 B 可证伪里程碑（第 8 节）量化阈值。"""

    learn_slope_max: float = 0.0        # 能力上升：任务内误差 EMA 斜率 < 0
    forget_rel_max: float = 0.25        # 免遗忘：带记忆时旧任务误差相对回升 <= 25%
    retain_gain_min: float = 0.10       # 记忆增益：保留误差相对无记忆改善 >= 10%
    combo_gain_min: float = 0.20        # 组合嵌入相对随机条件的零样本增益 >= 20%


# ----------------------------------------------------------------------
# 阶段 C（软件版内在化）：结构稀疏核心 + 能耗记账 + 能力复验
# 对应 docs/SRPC_DESIGN.md 不变量 3（能量内生·结构稀疏）与 6.1b（大模型能效对照）；
# 硬件部署本体（事件驱动/低比特芯片）deferred 至有专用硬件时
# ----------------------------------------------------------------------

@dataclass
class PhaseCConfig:
    """阶段 C 实验协议（无专用硬件，CPU/GPU 软件验证）。

    核心：出生即结构稀疏（扇入受限分块权重 + k-WTA）的核心从头重训，
    复跑阶段 B 全部验收（能力无回撤 = 低功耗来自架构本身，非稠密裁剪），
    并以硬件无关的有效 MAC 记账对比大模型标尺。
    """

    yardstick_params: tuple = (0.5e9, 1.5e9, 7e9)   # 大模型标尺参数量（6.1b 对照线）
    yardstick_names: tuple = ("LLM-0.5B", "LLM-1.5B", "LLM-7B")
    tokens_per_task: int = 512      # 大模型单任务推理 token 数（网格序列化+指令+输出）
    quant_bits: int = 8             # 低比特部署就绪检查（训练后权重量化）


@dataclass
class AcceptanceCConfig:
    """阶段 C 可证伪里程碑（第 8 节 C 行，软件版）量化阈值。"""

    llm_ratio_min: float = 1e3          # C2：每样本推理 MACs 比最小标尺低 >= 10^3 倍
    density_max: float = 0.35           # C3：权重总密度上限（出生即稀疏，由构造保证）
    quant_err_ratio_max: float = 1.5    # C4：int8 量化后冻结误差 <= fp 的 1.5 倍
    # C1 能力无回撤 = 稀疏核心上阶段 B 四项验收全部复现（复用 AcceptanceBConfig）


# ----------------------------------------------------------------------
# 阶段 E1（语言化起步）：字节级 UTF-8 词元前端 + 语言长程信用分配小任务
# 对应 docs/ROADMAP.md 阶段 E1 / docs/SRPC_DESIGN.md §9-2（自建词元前端）
# ----------------------------------------------------------------------

@dataclass
class LangConfig:
    """E1 语言长程信用分配（延迟文本关联）任务与阈值（与 §8.5 早筛同谱系）。

    任务族（字母表 A=16 个 ASCII 字母，流经 ByteTokenizer 的 256 维字节一热，
    与 ARC 符号编码 / E2 语言流同前端）：
    - **assoc（延迟关联，主验收任务）**：y_t = π(x_{t-Δ})，π = 固定随机置换。
      π 为双射且流均匀 => y 边际均匀，与任何单一输入符号**零边际相关**
      （同 §8.5 判据）——纯相关 Hebbian 一阶统计无信号，只有误差驱动的
      条件结构学习可解；同时确定性映射信号强，在线单样本可学。
    - **xorsum（延迟成对 XOR，边界记录）**：y_t = x_{t-Δ-1} ⊕ x_{t-Δ}
      （4 bit 逐位 XOR，16 类）。在线单样本学习**不可达**（v2 探针证据链，
      见下），降为在线信用分配边界的记录项，不计入验收。

    **v2 方案调整（2026-09-07，边界探针证据链）**：
    xorsum@16类 在线失败后逐层排查——(1) 超参全排除（credit 同款超参 /
    深收敛 64 iters / 学习率 0.05-1.0 / k-WTA / 掩码 / 事件门控 / 感受野
    窗口 2-3 块均无改善）；(2) **BP 在线对照同样失败**（同形状网络 +
    Adam 单样本 20k 步 ≈ 机会）——parity 为 SQ-hard 高频函数（Kearns &
    Valiant 1989; Blum et al. 1994），在线更新的期望梯度≈0，对称无法
    破缺；(3) **batch 上界 = 1.0**（mini-batch plain SGD 300 epoch 即达）
    ——任务本身可解；(4) PCN+梯度累积/经验回放均失败（k-WTA/clip 等
    非标准组件破坏收敛态≈BP 的等效性）。结论：瓶颈 = 在线协议的样本
    效率，非局部规则原理缺陷；修复路径 = F 阶段记忆回放（慢记忆 ->
    batch 等效协议，生物学对应睡眠重放），不在 E1 解决。
    assoc@Δ=16 亦为边界（远端块表征稀释：层1每单元锚定 1 块，1/17
    覆盖远端，容量不足，非原理性——扩 h1 或感受野可推远）。

    窗口 = 最近 W 步字节一热拼接（延迟线属机械外围序列化，§2.2）；
    W = Δ+1（assoc）/ Δ+2（xorsum）。远端块（窗口最前 1/2 块）为唯一
    任务相关块。网络与 CreditPCN 同一套 7.3 局部规则，多类化
    （x0 -> x1(时间分块感受野) -> x2(随机扇入) -> x3(C 类 one-hot)）。

    阈值预注册（v2，主任务 assoc，Δ=4 主判定）：
    acc_err >= 0.80、acc_hebb <= 0.20（机会 1/16=0.0625）、
    gap >= 0.40、远端块权重变化占比 >= 0.25（机会 1/(Δ+1)=0.20）；
    Δ=8 跨度外推 acc_err >= 0.80。Δ=1 近程对照为记录项。
    """

    # 任务几何
    alphabet: str = "abcdefghijklmnop"   # 16 字母（4 bit 码），i.i.d. 均匀流
    n_classes: int = 16
    delta: int = 4                       # 延迟跨度（与 §8.5 早筛 Δ=4 同起点）
    # 网络（CreditConfig 同谱系；按 256 维字节块放大隐层）
    h1: int = 128                        # L1 隐层（时间分块感受野，每单元锚定一个时间块）
    h2: int = 64                         # L2 隐层（随机扇入，联合特征层）
    x_max: float = 5.0
    settle_iters: int = 24               # err 臂推断收敛迭代（信用回传深度）
    hebb_settle_iters: int = 8           # hebb 臂浅迭代（纯相关瞬时联想）
    eta_inf: float = 0.09                # 推断阻尼步长（块正交掩码下谱半径 ~1）
    alpha: float = 1.5                   # 顶层类拉动 >> 底层重建（同 CreditConfig v2）
    beta: float = 1.0
    theta_event: float = 0.01
    eta_out: float = 0.1                 # 自由输出模式 x3 推断步长
    # 7.3 规则 2
    eta_w: float = 0.05
    theta_syn: float = 1e-2
    # 不变量 3：出生即结构稀疏
    fan_in_frac: float = 0.75
    kwta_frac: float = 0.5
    kwta_on: bool = True
    hebb_free: bool = True               # hebb 臂自由推断训练（消除类泄漏）
    energy_mode: str = "class"           # 钳制-比较能量口径（e1+e2，§8.5 同理）
    # 协议
    train_steps: int = 4000
    eval_steps: int = 600                # 自由推断评估（主口径：acc + BPC）
    compare_steps: int = 120             # 钳制-比较子样本（协议交叉验证）
    # 验收阈值（v2 预注册，主任务 assoc，Δ=4 主判定）
    acc_assoc_min: float = 0.80
    acc_assoc_d8_min: float = 0.80       # Δ=8 跨度外推
    acc_hebb_max: float = 0.20           # 纯相关对照上限（机会 1/16）
    gap_min: float = 0.40                # 误差驱动 - 纯相关
    distal_assoc_min: float = 0.25       # 远端驱动（机会 1/(Δ+1)=0.20）
    # 边界记录项（不设验收阈值）：xorsum@{4,16}（在线组合信用分配边界，
    # SQ-hard + BP 在线对照证据，修复排 F 阶段记忆回放）、assoc@16（远端
    # 表征稀释边界，容量非原理性）。

