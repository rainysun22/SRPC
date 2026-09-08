"""E2 登顶跑编排 driver：顺序确保 5 档 PCN(全 epoch) + 5 档孪生完成，再生成裁决报告。

- 幂等：PCN 档位完成（json curve 末点 step==steps）则跳过，否则 --resume 续跑；
  孪生 json 缺失则重跑。
- 顺序执行（单 GPU / RAM 受限，1660Ti + ~2GB 空闲 RAM 不支持高并发安全）。
用法（4032 档若已单独在跑，等其结束后执行；或先停后直接跑本 driver）：
    python scripts/run_e2_summit_all.py [--pcn-only] [--report-only]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results_e2_gpu")
H_LIST = [768, 1200, 1856, 2832, 4032]
PY = sys.executable


def _done(json_path: str, steps: int) -> bool:
    if not os.path.exists(json_path):
        return False
    with open(json_path) as f:
        d = json.load(f)
    want = d.get("steps", steps)          # json 自记录目标步数（真源）
    curve = d.get("curve", [])
    return bool(curve) and curve[-1]["step"] >= want


def run(args: list[str]) -> None:
    print("+ python " + " ".join(args), flush=True)
    r = subprocess.run([PY, *args], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(args)} (rc={r.returncode})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcn-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    steps = 1003838  # len(train)-W-1（tinyshakespeare 90% 段）

    if not args.report_only:
        # GPU 稀缺资源先占：5 档 PCN 全 epoch（每档幂等/续跑）
        for h in H_LIST:
            jp = os.path.join(RES, f"pcn_{h}.json")
            if not _done(jp, steps):
                run(["scripts/run_e2_summit.py", "--kind", "pcn", "--h",
                     str(h), "--device", "cuda", "--resume",
                     "--eval-every", "30000"])
        if not args.pcn_only:
            # 孪生（CPU，标准反传参照）
            for h in H_LIST:
                jp = os.path.join(RES, f"twin_{h}.json")
                if not _done(jp, steps):
                    run(["scripts/run_e2_summit.py", "--kind", "twin",
                         "--h", str(h)])
    # 报告
    run(["scripts/build_e2_summit_report.py"])
    print("== summit pipeline done ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
