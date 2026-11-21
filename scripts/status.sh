#!/bin/bash
echo "=== Processes ==="
ps aux | grep -E 'train.py|evaluate.py' | grep -v grep || echo 'No training process found'
echo
echo "=== Log tail ==="
tail -n 30 /root/DIT5411/train_all.log
echo
echo "=== Checkpoints ==="
ls -lh /root/DIT5411/models 2>/dev/null || true
ls -lh /root/DIT5411/results 2>/dev/null || true
