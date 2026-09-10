#!/bin/bash
# H2 有界验证：判别式 PC（dpc_amp）扫描 @ 1856 / 90k，并行 3 档，后台 detached。
cd /root/srpc_e2/srpc_src
for amp in 0.5 1.0 2.0; do
  nohup env PYTHONPATH=/root/srpc_e2/srpc_src python3 -B scripts/h2_dpc_validate.py 1856 90000 $amp 30000 \
    > logs/h2_dpc_amp${amp}.log 2>&1 &
  echo "launched amp=$amp pid=$!"
done
echo "all launched"