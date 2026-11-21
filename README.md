# DIT5411 Traditional Chinese Handwriting Recognition

A TensorFlow project for recognizing handwritten Traditional Chinese characters, built for the **THEi DIT5411 Machine Learning** assignment.

## Objective

Train a multi-layer neural network to classify up to **13,065** Traditional Chinese characters. The dataset is split so the first **40 samples per character** are used for training and the remaining samples are reserved for testing. Data augmentation is applied to expand the training set to at least **200 samples per character**.

## Dataset

This project uses the [Traditional Chinese Handwriting Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset).

- **Full dataset (13,065 characters, 300×300):**
  ```bash
  git clone https://github.com/chenkenanalytic/handwritting_data_all.git
  cat handwritting_data_all/all_data.zip* > all_data.zip
  unzip -O big5 all_data.zip -d data/raw
  ```
  The extracted folder should be `data/raw/cleaned_data/<character>/<character>_<id>.png`.

- **Common words dataset (4,803 characters, 50×50):**
  Download the four zip files from the AI-FREE-Team repository, extract them into `data/raw/`. The folder will be named `cleaned_data(50_50)`.

## Project Structure

```text
DIT5411/
├── config.py                 # Hyperparameters and paths
├── prepare_data.py           # Split dataset into train / test
├── augment.py                # Generate augmented training images
├── models.py                 # CNN architectures (A, B, C)
├── train.py                  # Train a selected model
├── evaluate.py               # Evaluate and compare models
├── visualize_history.py      # Plot training curves and comparison charts
├── utils.py                  # Helper functions
├── requirements.txt          # Python dependencies
├── notebooks/                # Google Colab notebook
├── data/                     # Dataset, train/test splits
├── models/                   # Saved model checkpoints
└── results/                  # Training history and comparison tables
```

## Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Prepare the data

After placing the raw dataset under `data/raw/`, run:

```bash
python prepare_data.py
```

This creates `data/train/` and `data/test/` with the first 40 samples per class in the training folder.

> **Note:** The original dataset uses Chinese-character folder and file names. To avoid encoding issues with OpenCV and TensorFlow on Windows, `prepare_data.py` renames folders and files to numeric IDs and stores the mapping in `results/metadata.json`.

### 2. Augment the training data (optional)

On-the-fly augmentation is enabled by default in `train.py`. If you prefer to generate explicit augmented files, run:

```bash
python augment.py --factor 5
```

This creates 5 augmented copies per training image, giving 40 × 5 = 200 training samples per class.

### 3. Train a model

```bash
python train.py --model a --epochs 5 --batch-size 1024
```

Available models:

| Model | Description |
|-------|-------------|
| `a`   | Deep CNN (32 → 64 → 128 filters, 128 dense units) |
| `b`   | Medium CNN (16 → 32 filters, 64 dense units) |
| `c`   | Small CNN (8 filters, 16 dense units) |

### 4. Evaluate

Evaluate a single model:

```bash
python evaluate.py --model a
```

Compare all three models:

```bash
python evaluate.py --compare
```

### 5. Visualize results

Plot training curves:

```bash
python visualize_history.py --model a
```

Plot model comparison:

```bash
python visualize_history.py --compare
```

## Model Architectures

All models receive grayscale images resized to `48 × 48`, use `LeakyReLU` activation, `BatchNormalization` and `Dropout`, and end with a `Dense(NUM_CLASSES, activation='softmax')` output layer.

## Results

After training, results are saved under `results/`:

- `metadata.json` — dataset statistics and label mapping
- `history_<model>.json` — training history
- `test_results_<model>.json` — test loss, top-1 and top-5 accuracy
- `model_comparison.json` — side-by-side comparison of all models

You can update this section with your own numbers after running the experiments.

## Google Colab

Open `notebooks/DIT5411_Chinese_Recognition_Colab.ipynb` and run the cells to train on Google Colab with a GPU.

## References

- Traditional Chinese Handwriting Dataset: https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset
- Full 13,065-character dataset: https://github.com/chenkenanalytic/handwritting_data_all
- OpenCV augmentation tutorial: https://github.com/darshitajain/Introduction-to-OpenCV
- Handwritten Digit Recognition with Keras and TensorFlow: https://github.com/kimanalytics/Handwritten-Digit-Recognition-using-Keras-and-TensorFlow

## Author

YOUR_NAME — DIT5411 Machine Learning Assignment
