"""E3 语料：顺序多语字节流构造器（语言版记忆价值验证的冒烟载体）。

设计目标（对应 /workspace/e3/README_SMOKE.md 的"语言性"论证）：
  1) **顺序多语流**：K=3 种语种/风格字节流（latin 英文 / cyrillic 俄文 / cjk 中文），
     全部经 UTF-8 编码为 uint8 字节数组。训练段按 lang0→lang1→lang2 顺序拼接成大
     字节流（顺序学习协议，与 Phase B 免遗忘同一协议族）。
  2) **窗口不足**：三种语种**共享同一套 ASCII 空白/标点**（空格 / 换行 / 逗号 / 句号），
     且每段段首都加了共享 ASCII 前缀 PROLOGUE。因此在语种边界处 / 段首处，
     单个 16 字节窗口（=E2Config.context）内的字节**不足以唯一确定当前语种**
     （一堆 ASCII 空格换行既可来自 latin 也可来自 cyrillic 或 cjk）。
  3) **全局统计可分**：语种的**整段字节频率**（原型）可区分——cyrillic 的高位
     UTF-8 连续字节落在 0xD0-0xD1、cjk 落在 0xE0 系三字节区段、latin 主要是
     0x20-0x7E 单字节。因此"整段语种字节统计"能超过单窗口提供 top-down 先验。

生成方式（全部确定性子程序，seed 可复现）：每语种一个小词表，i.i.d. 采样拼接
成词流（空格分隔，偶尔换行），目标 = 下一字节。词流内容在 train/eval 用不同
seed，保证评估段内容不曾在训练中见过（泛化口径，与 E1/E2 同协议族）。

对外接口：
    LANGS = ["latin", "cyrillic", "cjk"]
    build_train_segments(seed) -> [(lang, np.uint8 bytes), ...]   # 每语种一大段（顺序训练）
    build_eval_stream(seed, reps) -> dict                          # 含边界的评估样本
"""
from __future__ import annotations

import numpy as np

LANGS = ["latin", "cyrillic", "cjk"]

# 共享 ASCII 标点/空白（所有语种共用 → 语种间共享字节，制造"窗口不足"）
_PROLOGUE = "\n\n "          # 段首共享 ASCII 前缀（窗口内无法判语种的典型位置）
_PUNCT_COMMON = " .,"

# ---- 语种字元集合（字符层；UTF-8 编码后的高位字节区分语种）----
_LATIN_CHARS = list("abcdefghijklmnopqrstuvwxyz")
_CYRILLIC_CHARS = list("абвгдежзийклмнопрстуфхцчшщъыьэюя")
_CJK_CHARS = list(
    "的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年"
    "得就那要下以生会自着去之过家学对可她里后小么心多天而能好都然"
    "没日于起还发成事只作当想看见面又主像把美本明问力理合"

)
_CHARSET = {"latin": _LATIN_CHARS, "cyrillic": _CYRILLIC_CHARS,
            "cjk": _CJK_CHARS}

# ---- 语种词表（短词，i.i.d. 采样）----
_WORD_LATIN = [
    "the", "of", "and", "in", "that", "for", "with", "on", "at", "by",
    "from", "is", "it", "this", "be", "as", "will", "which", "have",
    "all", "cell", "mind", "tree", "water", "light", "cloud", "stone",
    "river", "fire", "wind", "forest", "bird", "path", "dream",
]
_WORD_CYR = [
    "бы", "и", "в", "не", "на", "что", "с", "он", "как", "это", "по",
    "для", "от", "она", "мы", "вы", "они", "когда", "если", "его",
    "река", "лес", "камень", "огонь", "ветер", "облако", "птица",
    "дорога", "мечта", "вода", "свет", "горы", "зима", "море",
]
_WORD_CJK = [
    "的", "一", "是", "不", "了", "人", "我", "在", "有", "他", "这",
    "大", "来", "上", "国", "到", "说", "们", "为", "子", "和", "你",
    "地", "出", "道", "时", "年", "得", "那", "就", "要", "下", "以",
    "生", "会", "自", "着", "去", "过", "家", "学", "对", "她", "里",
    "后", "小", "么", "心", "多", "天", "日", "于", "起", "还", "发",
    "成", "事", "只", "作", "当", "想", "看", "见", "面", "又", "主",
    "像", "把", "美", "本", "明", "问", "力", "理", "合", "月亮", "森林",
]
_WORD_VOCAB = {"latin": _WORD_LATIN, "cyrillic": _WORD_CYR, "cjk": _WORD_CJK}


def _utf8_bytes(s: str) -> np.ndarray:
    return np.frombuffer(s.encode("utf-8"), dtype=np.uint8)


def _sample_words(lang: str, rng: np.random.Generator, n_words: int) -> str:
    vocab = _WORD_VOCAB[lang]
    words = [vocab[int(rng.integers(0, len(vocab)))] for _ in range(n_words)]
    # 句子级换行（共享 ASCII 换行/空格，跨语种共享字节），其余空格分隔
    s = words[0]
    for w in words[1:]:
        sep = "\n " if rng.random() < 0.05 else " "
        s += sep + w
    return s


def _seg_words_until(lang: str, rng: np.random.Generator,
                     min_bytes: int, max_words: int) -> str:
    """采样词直到该语种 UTF-8 字节长度 >= min_bytes（或达到 max_words）。"""
    acc, n = "", 0
    while n < max_words and len(acc.encode("utf-8")) < min_bytes:
        vocab = _WORD_VOCAB[lang]
        w = vocab[int(rng.integers(0, len(vocab)))]
        sep = "\n " if (n and rng.random() < 0.05) else (" " if n else "")
        acc += sep + w
        n += 1
    return acc


# ----------------------------------------------------------------------
# 训练段：每语种一大段（顺序学习 lang0→lang1→lang2）
# ----------------------------------------------------------------------
def build_train_segments(seed: int = 0, min_bytes: int = 4200,
                         max_words: int = 3000) -> list:
    """返回 [(lang, np.uint8 字节数组), ...]，按 LANGS 顺序（lang0→lang1→lang2）。

    每段 = PROLOGUE + 语种词流。段首共享 ASCII 前缀使"单窗口判语种"在部分
    位置上确实不充分，但整段的语种字节统计可分。
    """
    rng = np.random.default_rng(1000 + seed)   # 训练流 seed（独立于 eval）
    segs = []
    for lang in LANGS:
        s = _PROLOGUE + _seg_words_until(lang, rng, min_bytes, max_words)
        segs.append((lang, _utf8_bytes(s)))
    return segs


# ----------------------------------------------------------------------
# 评估流：含语种切换边界的顺序流（多段、多边界）+ 一热窗口样本
# ----------------------------------------------------------------------
def build_eval_stream(seed: int = 0, reps: int = 3, min_bytes: int = 90,
                      tail_win: int | None = None) -> dict:
    """构造含边界的评估字节流与一热窗口样本（窗口不足的段首 / 边界集中考察）。

    返回 dict：
        X        (n, W, 256) float32 一热窗口样本
        y        (n,) int64        目标字节
        groups   (n,) int64        每样本所属语种（truth label，评估时用于 oracle recall）
        switch   (n,) bool         是否"切换尾"样本（段首附近 tail_win 字节内）
        stream   (N,) uint8        评估字节流整体（供诊断）
        boundaries (list)          (global_pos, lang) 各段起点
        说明：评估内容用独立 seed 生成的语种词流（train 未见），泛化口径。
    """
    rng = np.random.default_rng(2000 + seed)   # 评估流 seed（独立）
    if tail_win is None:
        tail_win = 16                        # 默认 = W = context
    # ---- 生成 eval 分段（round-robin 保证每语种多次出现、多边界）----
    seg_bytes = []   # 每段 [[...bytes], ...] 先存（需已知段长以便窗口回看）
    seg_lang = []    # 每段语种
    for r in range(reps):
        for lang in LANGS:
            s = _PROLOGUE + _seg_words_until(lang, rng, min_bytes,
                                             max_words=200)
            seg_bytes.append(list(_utf8_bytes(s)))
            seg_lang.append(lang)
    # ---- 拼成全局流，记录段起点与逐字节语种 ----
    stream = []
    boundaries = []            # (global_pos, lang_index)
    lang_pos = []              # 逐字节语种 index
    _idx = {l: i for i, l in enumerate(LANGS)}
    for b, l in zip(seg_bytes, seg_lang):
        boundaries.append((len(stream), _idx[l]))
        lang_pos.extend([_idx[l]] * len(b))
        stream.extend(b)
    stream = np.array(stream, np.uint8)
    lang_pos = np.array(lang_pos, np.int64)
    W = 16

    # ---- 候选窗口位点：target t 满足有前 W 字节（t>=W）----
    # switch（切换尾）位点：目标 t 距所在段起点 offset = t - bp < tail_win，
    # 即窗口跨段首 / 落在段首共享 ASCII 前缀附近 → 单窗口不足以判语种。
    sw = np.zeros(len(stream), bool)
    for bp, _ in boundaries:
        sw[bp:min(bp + tail_win, len(stream))] = True
    valid = np.arange(W, len(stream))          # 合法目标位点（t>=W）
    switch_v = sw[valid]

    # 抽样：switch 位点全保留（stride 1），其余粗抽样（stride）控制样本量
    stride = 2
    keep = switch_v | (np.arange(len(valid)) % stride == 0)
    tsel = valid[keep]
    swsel = switch_v[keep]

    # 构建一热窗口
    n = len(tsel)
    X = np.zeros((n, W, 256), np.float32)
    y = np.zeros(n, np.int64)
    groups = np.zeros(n, np.int64)
    for i, t in enumerate(tsel):
        X[i, np.arange(W), stream[t - W:t]] = 1.0
        y[i] = stream[t]
        groups[i] = lang_pos[t]

    return dict(X=X, y=y, groups=groups, switch=swsel,
                stream=stream, boundaries=boundaries,
                tail_win=tail_win)


def train_segment_windows(seg: np.ndarray, W: int = 16):
    """顺序供给一个语种段的 (x0 一热, y, t) —— 监督训练/慢记忆巩固。"""
    n = len(seg) - W - 1
    for t in range(n):
        x = np.zeros((W, 256), np.float32)
        x[np.arange(W), seg[t:t + W]] = 1.0
        yield x, int(seg[t + W]), t


# ----------------------------------------------------------------------
# 语种可区分度的诊断工具（"全局统计可分"的证据）
# ----------------------------------------------------------------------
def lang_byte_hist(segs: list) -> dict:
    """由各语种训练段算 256 维字节直方图（全局语种统计原型）。"""
    h = {}
    for lang, b in segs:
        cnt = np.bincount(b, minlength=256).astype(np.float64)
        p = cnt / cnt.sum()
        h[lang] = p
    return h


if __name__ == "__main__":
    tr = build_train_segments()
    hist = lang_byte_hist(tr)
    for lang, p in hist.items():
        print(f"{lang}: len={len(tr[LANGS.index(lang)][1])} "
              f"bytes, top bytes={np.argsort(p)[-6:][::-1]}, "
              f"hair={p[0xd0:0xd2].sum():.3f}/{p[0xe4:0xe9].sum():.3f}")
    ev = build_eval_stream()
    print("eval windows:", ev["X"].shape, "switch frac:",
          round(ev["switch"].mean(), 3), "boundaries:", len(ev["boundaries"]))