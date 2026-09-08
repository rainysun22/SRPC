#!/bin/bash
# Detached launcher for the per-step W2-cap fresh validation run (4090).
cd /root/srpc_e2 || exit 1
pkill -f val_fix_retrain.py 2>/dev/null
sleep 1
rm -f val_fix_retrain_fresh2.log
setsid nohup /root/srpc_e2/venv/bin/python -B /root/srpc_e2/val_fix_retrain.py fresh \
    </dev/null > val_fix_retrain_fresh2.log 2>&1 &
echo "LAUNCHED pid=$!" > /root/srpc_e2/launch_fresh.pid
exit 0