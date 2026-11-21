#!/bin/bash
set -e
cd /root/DIT5411
source venv/bin/activate

# Point TensorFlow to the CUDA libraries shipped with tensorflow[and-cuda]
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_cupti/lib:$SITE/nvidia/cuda_nvrtc/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/curand/lib:$SITE/nvidia/cusolver/lib:$SITE/nvidia/cusparse/lib:$SITE/nvidia/nccl/lib:$SITE/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

echo "LD_LIBRARY_PATH set"
python - <<'PY'
import os
import tensorflow as tf
print("TF", tf.__version__)
print("GPUs", tf.config.list_physical_devices("GPU"))
print("Built with CUDA:", tf.test.is_built_with_cuda())
PY
