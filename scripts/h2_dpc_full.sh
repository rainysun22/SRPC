#!/bin/bash
# H2 DPC 全程（300k）：amp=1.0 / 0.5 并行，后台 detached。
cd /root/srpc_e2/srpc_src
for amp in 0.5 1.0; do
  nohup env PYTHONPATH=/root/srpc_e2/srpc_src python3 -B scripts/h2_dpc_validate.py 1856 300000 $amp 30000 \
    > logs/h2_dpc30k_amp${amp}.log 2>&1 &
  echo "launched amp=$amp pid=$!"
done
echo "all launched"