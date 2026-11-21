#!/bin/bash
# Continue / retrain after architecture fixes.
# - Resume Model A to more epochs
# - Retrain Model B and C from scratch with improved architectures

set -e
cd /root/DIT5411
source venv/bin/activate

SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_cupti/lib:$SITE/nvidia/cuda_nvrtc/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/curand/lib:$SITE/nvidia/cusolver/lib:$SITE/nvidia/cusparse/lib:$SITE/nvidia/nccl/lib:$SITE/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export TF_CPP_MIN_LOG_LEVEL=1

cp /mnt/c/Users/ihenr/Downloads/DIT5411/models.py /mnt/c/Users/ihenr/Downloads/DIT5411/config.py /mnt/c/Users/ihenr/Downloads/DIT5411/utils.py /mnt/c/Users/ihenr/Downloads/DIT5411/train.py /root/DIT5411/

# Wait until current train.py exits
while pgrep -f 'train.py' >/dev/null; do
  echo "Waiting for current training to finish..."
  sleep 30
done

echo "===== Resume Model A to 10 epochs ====="
python train.py --model a --epochs 10 --batch-size 256 --resume

echo "===== Retrain Model B ====="
rm -f models/best_model_b.keras models/final_model_b.keras results/history_b.json
python train.py --model b --epochs 5 --batch-size 256

echo "===== Retrain Model C ====="
rm -f models/best_model_c.keras models/final_model_c.keras results/history_c.json
python train.py --model c --epochs 5 --batch-size 256

echo "===== Evaluate ====="
python evaluate.py --compare
python visualize_history.py --compare
echo "===== ALL DONE ====="
