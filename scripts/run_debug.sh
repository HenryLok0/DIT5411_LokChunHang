#!/bin/bash
set -e
cd /root/DIT5411
source venv/bin/activate
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_cupti/lib:$SITE/nvidia/cuda_nvrtc/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/curand/lib:$SITE/nvidia/cusolver/lib:$SITE/nvidia/cusparse/lib:$SITE/nvidia/nccl/lib:$SITE/nvidia/nvjitlink/lib"
cp /mnt/c/Users/ihenr/Downloads/DIT5411/scripts/debug_overfit.py /root/DIT5411/
sed -i 's/\r$//' /root/DIT5411/debug_overfit.py
python debug_overfit.py
