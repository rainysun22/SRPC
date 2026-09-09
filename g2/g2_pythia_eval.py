"""G2b：pythia 同规模 LLM 对照 —— GLUE RTE few-shot 推理评测。

SRPC 指（G2a 槽-HRR 组合记忆）在离散颜色关系 ICL 上的 acc 曲线，本脚本给出
pythia-14m/31m 在 few-shot *语言*推理（GLUE RTE entailment 二分类）上的 acc，
两者作为 G2 报告的同规模对照（诚实标注介质差异：离散符号推理 vs 自然语言推理）。

用法（4090 / CUDA）：
  python3 g2_pythia_eval.py --models 14,31 --ks 0,2,4,8 \
      --seed 0 --split_size 256 --out g2/results_g2_pythia_rte.json

细节：
  - 模板（prompt）：<n 个演示> + 测试项，答案 verbalizer = yes(entailment)/no(not-entailment)。
  - 打分：对(批量)最后一步 logits，比较候选 token " yes"/" no" 的对数概率 → 二分类 acc。
  - few-shot 演示从训练集固定种子抽样；K=0 为 0-shot。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

TEMPLATE = (
    "premise: {premise}\n"
    "hypothesis: {hypothesis}\n"
    "Does the hypothesis follow from the premise?\n"
    "Yes/No: "
)
LABEL_WORDS = ["yes", "no"]          # [entailment, not_entailment]
VERB_YES, VERB_NO = LABEL_WORDS


def build_demo(item, label) -> str:
    return TEMPLATE.format(premise=item["sentence1"], hypothesis=item["sentence2"]) + LABEL_WORDS[label]


def encode_candidates(tok) -> tuple[int, int] | None:
    """取候选词成单 token id；返回 (yes_id, no_id) 或 None(若拆词)。"""
    # 简化为：使用 " yes" / " no"（带前导空格）的首 token 作为候选，取其中词表存在者
    y = tok(" " + VERB_YES, add_special_tokens=False)["input_ids"]
    n = tok(" " + VERB_NO, add_special_tokens=False)["input_ids"]
    if len(y) == 1 and len(n) == 1:
        return int(y[0]), int(n[0])
    y2 = tok(VERB_YES, add_special_tokens=False)["input_ids"]
    n2 = tok(VERB_NO, add_special_tokens=False)["input_ids"]
    if len(y2) == 1 and len(n2) == 1:
        return int(y2[0]), int(n2[0])
    return None


def infer_acc(model, tok, valid, k, demos, device, bs, cand):
    yes_id, no_id = cand
    total = hits = 0
    # 预构建所有 prompt -> (input_ids, label)
    batch_in, batch_lab = [], []
    for item in valid:
        prompt = ""
        for i in range(min(k, len(demos))):
            d = demos[i]
            prompt += build_demo(d, d["label"]) + "\n"
        prompt += TEMPLATE.format(premise=item["sentence1"], hypothesis=item["sentence2"])
        enc = tok(prompt, return_tensors="pt")
        batch_in.append(enc["input_ids"][0])
        batch_lab.append(int(item["label"]))
    # 统一长度 padding（同 batch），逐批前向
    maxlen = max(len(x) for x in batch_in)
    for st in range(0, len(batch_in), bs):
        chunk = batch_in[st:st + bs]
        L = [len(x) for x in chunk]
        pad = torch.full((len(chunk), maxlen), tok.pad_token_id or 0, dtype=torch.long)
        for i, x in enumerate(chunk):
            pad[i, :len(x)] = x
        am = (pad != (tok.pad_token_id or 0)).long()
        with torch.no_grad():
            lg = model(pad.to(device), attention_mask=am.to(device)).logits
        rowv = {}
        for j in range(len(chunk)):
            last = lg[j, L[j] - 1]
            py = float(last[yes_id]); pn = float(last[no_id])
            pred = 0 if py >= pn else 1
            true = int(batch_lab[st + j])
            total += 1
            hits += int(pred == true)
    return hits / max(total, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="14,31")
    ap.add_argument("--ks", default="0,2,4,8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_size", type=int, default=256)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", default="/workspace/g2/results_g2_pythia_rte.json")
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]
    ks = [int(x) for x in args.ks.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    ds = load_dataset("nyu-mll/glue", "rte")
    train = ds["train"]                     # 演示池
    valid = list(ds["validation"])
    rng.shuffle(valid)
    valid = valid[:args.split_size]
    demos_pool = list(train)
    rng.shuffle(demos_pool)

    out = {"models": {}, "template": TEMPLATE.splitlines()[0], "seed": args.seed}
    for mn in models:
        name = f"EleutherAI/pythia-{mn}m-deduped"
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16 if dev == "cuda" else torch.float32
        ).to(dev)
        model.eval()
        cand = encode_candidates(tok)
        res = {"k_list": ks}
        for k in ks:
            demos = demos_pool[:k]
            t0 = time.time()
            acc = infer_acc(model, tok, valid, k, demos, dev, args.bs, cand)
            res[f"acc_k{k}"] = round(acc, 4)
            res[f"wall_k{k}"] = round(time.time() - t0, 1)
            print(f"  pythia-{mn}m  k={k}  acc={acc:.4f}  ({time.time()-t0:.0f}s)")
        out["models"][f"pythia-{mn}m"] = res
        del model
        torch.cuda.empty_cache() if dev == "cuda" else None

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out, "| device =", dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())