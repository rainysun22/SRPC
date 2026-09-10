#!/bin/bash
# H2 有界验证：自由沉降输出误差 PC（free_nudge）扫描 @ 1856 / 90k。
# 并行 3 档 nudge，后台 detached，结果写入 results_e2_gpu_eta/h2_free_*.json
cd /root/srpc_e2/srpc_src
for nu in 0.2 0.4 0.7; do
  nohup env PYTHONPATH=/root/srpc_e2/srpc_src python3 -B scripts/h2_free_validate.py 1856 90000 $nu 30000 \
    > logs/h2_free_nu${nu}.log 2>&1 &
  echo "launched nu=$nu pid=$!"
done
echo "all launched"