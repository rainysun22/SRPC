#!/bin/bash
# H2 充分监督 PC（deep-DPC）门控扫描 @1856/90k：amp=0.5 fl=12 下沉 dpc_deep∈{0.5,1.0,2.0}
cd /root/srpc_e2/srpc_src
for df in 0.5 1.0 2.0; do
  nohup env PYTHONPATH=/root/srpc_e2/srpc_src python3 -B scripts/h2_dpc_validate.py 1856 90000 0.5 30000 12 $df \
    > logs/h2_dpcd_df${df}.log 2>&1 &
  echo "launched df=$df pid=$!"
done
echo all launched