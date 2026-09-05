#!/bin/bash
export LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/hyhal/lib:$LD_LIBRARY_PATH
cd /root/kv
python3 run_qwen38_validate.py --tag v1 2>&1
