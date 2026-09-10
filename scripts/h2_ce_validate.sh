#!/bin/bash
# H2 有界验证扫描（串行）：ce_amp ∈ {0, 0.05, 0.2, 1.0} x 300k 步 1856。
cd /root/srpc_e2/srpc_src || exit 1
mkdir -p results_e2_gpu_eta
for amp in 0 0.05 0.2 1.0; do
  echo "===== RUN amp=$amp $(date) ====="
  python -B scripts/h2_ce_validate.py 1856 300000 "$amp" \
    > "results_e2_gpu_eta/h2_ce_1856_amp${amp}.log" 2>&1
  echo "===== DONE amp=$amp rc=$? $(date) ====="
done
echo "ALL-DONE"