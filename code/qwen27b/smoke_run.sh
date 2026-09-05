#!/bin/bash
export LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/hyhal/lib:$LD_LIBRARY_PATH
cd /root/kv
python3 /root/kv/smoke_qwen38.py 2>&1 | tail -40
