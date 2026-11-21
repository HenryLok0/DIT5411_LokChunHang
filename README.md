# DIT5411 Traditional Chinese Handwriting Recognition

This THEi Machine Learning assignment compares three TensorFlow/Keras convolutional neural networks on the full Traditional Chinese Handwriting Dataset. Given one grayscale image, each full model outputs a probability distribution over **13,065 characters**.

## Data and experimental method

The official archive contains **684,677 images**. Each source folder has a numeric ID from 1 to 13,065; the actual character is stored in the filenames. The project reads the legacy CP950 ZIP names directly, sorts each class by **numeric sample ID**, uses the first 40 originals for training and reserves all later originals for testing.

| Split or coverage | Count |
|---|---:|
| Original training images | 522,127 |
| Untouched test images | 162,550 |
| Character classes represented in the test set | 13,022 |
| Training classes without any test images | 43 |
| Augmented training draws per epoch | 2,613,000 |

The 43 short classes have 16–37 originals. All their originals go to training, and each still gets its own output neuron. No synthetic test images are generated, so the test scores don't measure these 43 classes. The test set is never opened during training.

### Preprocessing and augmentation

Images are decoded as grayscale, inverted if the border indicates a dark background, area-resized from 300×300 to 48×48, and stored as compact byte arrays. Pixel values are normalized to [0, 1] when batching. The full cache uses about 1.6 GB without extracting hundreds of thousands of separate files.

For a class with 40 originals, every original appears as **five independently sampled affine draws every epoch**. For classes with fewer originals, the input sampler cycles over the originals until it reaches the same 200 draws. Keras applies rotation up to ±10°, zoom (±12% sampling-window change), and horizontal shear (factor ±0.08), with white fill. These are new random draws, not a claim that all transformed byte arrays are unique. All augmentation layers are inactive during evaluation.

### CNN comparison

| Model | Convolution filters by block | Dense units | Dropout | Full output |
|---|---|---:|---:|---:|
| A (deep) | 32 → 64 → 128 | 256 | 0.30 | 13,065 |
| B (medium) | 24 → 48 | 128 | 0.25 | 13,065 |
| C (small) | 16 | 64 | 0.20 | 13,065 |

Each model uses 3×3 convolutions, batch normalization, Leaky ReLU, and a 6×6 spatial grid before flattening into the dense classifier. Training uses Adam and sparse categorical cross-entropy with the same split, batch size, seed and predetermined epoch budget for all models. Final checkpoints are compared on the same held-out images. Top-1 measures exact first-choice recognition; top-5 checks whether the correct class appears among the first five. The comparison reports the highest observed top-1 score, including ties.

## Run the full experiment

Use Python 3.11 or 3.12. A GPU is recommended: each epoch processes over 2.6 million training draws. If a batch does not fit GPU memory, reduce the batch size for all three models and start a fresh run directory.

```bash
python -m pip install -r requirements.txt
python download_data.py
python prepare_data.py --archive data/all_data.zip
python train.py --model a --epochs 5 --batch-size 128 --resume
python train.py --model b --epochs 5 --batch-size 128 --resume
python train.py --model c --epochs 5 --batch-size 128 --resume
python evaluate.py --report reports/full-results.md
python visualize_history.py
```

On Linux or WSL2 with a supported NVIDIA GPU, follow the [TensorFlow GPU installation guide](https://www.tensorflow.org/install/pip) if TensorFlow doesn't detect the GPU. You may pass `--mixed-precision` to all three training commands on a compatible GPU. Do not change options midway through a run. `--resume` continues at the beginning of the next committed epoch, restoring the saved model and optimizer. An interrupted partial epoch is rerun; resumed random augmentation is not guaranteed bit-for-bit identical.

The [Colab notebook](notebooks/DIT5411_Chinese_Recognition_Colab.ipynb) provides the same workflow with GPU checks and Google Drive persistence. Existing `data/train`, `data/test`, `models/`, and `results/` from the older workflow are not used or overwritten.

To try a completed model on a single-character image, run `python predict.py character.png --model a`, replacing `a` with the architecture you want to use. The output lists five characters and their model scores.

### Results

**Full-dataset accuracy and the winning architecture are pending the full experiment.** After all three runs finish, the evaluation command writes `reports/full-results.md` and `runs/full/model_comparison.json`. It rejects unfinished or mismatched runs. Link the generated full results in this README before submitting.

The [GitHub Actions experiment](../../actions/workflows/full-experiment.yml) runs the three architectures independently on standard CPU runners with the same three-epoch budget and batch size of 256. It publishes the comparison only after all three full-dataset evaluations pass validation.

The [32-class CPU smoke test](reports/smoke-results.md) checks the pipeline on real handwriting images. Its scores must not be reported as assignment results. The exact ZIP index counts are recorded in the [dataset audit](reports/dataset-audit.json).

To reproduce the smoke test without mixing it with full-run artifacts:

```bash
python prepare_data.py --archive data/all_data.zip --smoke
python train.py --model a --epochs 5 --batch-size 128 --resume --smoke
python train.py --model b --epochs 5 --batch-size 128 --resume --smoke
python train.py --model c --epochs 5 --batch-size 128 --resume --smoke
python evaluate.py --smoke
```

## References and dataset attribution

- [Traditional Chinese Handwriting Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset) by the AI . FREE Team, developed from STUST EECS_Chinese MNIST (總集). The [full 13,065-character archive](https://github.com/chenkenanalytic/handwritting_data_all) identifies the dataset terms as CC BY-NC-SA 4.0. Dataset images are not redistributed in this repository.
- [OpenCV affine transformation reference](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html) and the assignment's [Introduction-to-OpenCV example](https://github.com/darshitajain/Introduction-to-OpenCV).
- [Handwritten Digit Recognition using Keras and TensorFlow](https://github.com/kimanalytics/Handwritten-Digit-Recognition-using-Keras-and-TensorFlow), an assignment reference for project structure.
