#!/bin/bash
# 带修复(fix: per-step W2 cap=5.0)的顺序稳定性诊断：2832 与 4032 档，
# 各训练越过各自历史崩溃点(~33万/45万)一段健康余量，探 wsmax/自由推断 BPC 无发散。
cd /root/srpc_e2 || exit 1
rm -f fullb_diag.log
for tier in "2832 420000" "4032 560000"; do
  set -- $tier
  echo "[launcher] starting h=$1 steps=$2 @ $(date)" >> fullb_diag.log
  /root/srpc_e2/venv/bin/python -B /root/srpc_e2/fullb_fix.py "$1" "$2" 30000 >> fullb_diag.log 2>&1
  echo "[launcher] finished h=$1 rc=$? @ $(date)" >> fullb_diag.log
done
echo "[launcher] ALL DONE @ $(date)" >> fullb_diag.log