#!/bin/bash
# H2 DPC 门控扫描 @1856/90k：调 free_iters（沉降累积判别塑造）。后台并行。
cd /root/srpc_e2/srpc_src
for spec in "0.5 24" "0.5 48" "0.3 24" "0.4 24"; do
  set -- $spec; amp=$1; fi=$2
  nohup env PYTHONPATH=/root/srpc_e2/srpc_src python3 -B scripts/h2_dpc_validate.py 1856 90000 $amp 30000 $fi \
    > logs/h2_dpcg_amp${amp}_fi${fi}.log 2>&1 &
  echo "launched amp=$amp free_iters=$fi pid=$!"
done
echo "all launched"