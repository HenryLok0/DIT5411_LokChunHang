#!/bin/bash
set -e
cd /root/DIT5411
source venv/bin/activate

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_cupti/lib:$SITE/nvidia/cuda_nvrtc/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/curand/lib:$SITE/nvidia/cusolver/lib:$SITE/nvidia/cusparse/lib:$SITE/nvidia/nccl/lib:$SITE/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export TF_CPP_MIN_LOG_LEVEL=1
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/root/DIT5411/venv

EPOCHS=${1:-3}
BATCH=${2:-2048}

echo "Starting train-all: epochs=$EPOCHS batch=$BATCH"
echo "Log: /root/DIT5411/train_all.log"

nohup bash -c "
  source /root/DIT5411/venv/bin/activate
  export LD_LIBRARY_PATH='$LD_LIBRARY_PATH'
  export TF_CPP_MIN_LOG_LEVEL=1
  cd /root/DIT5411
  for m in a b c; do
    echo ===== Training model \$m =====
    python train.py --model \$m --epochs $EPOCHS --batch-size $BATCH --resume
  done
  echo ===== Evaluating =====
  python evaluate.py --compare
  python visualize_history.py --compare
  echo ===== ALL DONE =====
" > /root/DIT5411/train_all.log 2>&1 &

echo $! > /root/DIT5411/train_all.pid
echo "PID $(cat /root/DIT5411/train_all.pid)"
echo "Tail log with: wsl -d Ubuntu-22.04 -u root -- tail -f /root/DIT5411/train_all.log"
