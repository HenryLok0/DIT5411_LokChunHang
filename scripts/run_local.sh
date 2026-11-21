#!/bin/bash
# Run training inside WSL Ubuntu with TensorFlow GPU.
# Usage:
#   bash scripts/run_local.sh prepare
#   bash scripts/run_local.sh train a 3
#   bash scripts/run_local.sh train-all
#   bash scripts/run_local.sh evaluate

set -e
cd /root/DIT5411
source venv/bin/activate

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_cupti/lib:$SITE/nvidia/cuda_nvrtc/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/curand/lib:$SITE/nvidia/cusolver/lib:$SITE/nvidia/cusparse/lib:$SITE/nvidia/nccl/lib:$SITE/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export TF_CPP_MIN_LOG_LEVEL=1

CMD=${1:-help}

case "$CMD" in
  prepare)
    python prepare_data.py
    ;;
  train)
    MODEL=${2:-a}
    EPOCHS=${3:-3}
    BATCH=${4:-2048}
    python train.py --model "$MODEL" --epochs "$EPOCHS" --batch-size "$BATCH" --resume
    ;;
  train-all)
    EPOCHS=${2:-3}
    BATCH=${3:-2048}
    for m in a b c; do
      echo "===== Training model $m ====="
      python train.py --model "$m" --epochs "$EPOCHS" --batch-size "$BATCH" --resume
    done
    ;;
  evaluate)
    python evaluate.py --compare
    python visualize_history.py --compare
    ;;
  *)
    echo "Usage: $0 {prepare|train|train-all|evaluate}"
    echo "  prepare"
    echo "  train <a|b|c> [epochs] [batch]"
    echo "  train-all [epochs] [batch]"
    echo "  evaluate"
    ;;
esac
