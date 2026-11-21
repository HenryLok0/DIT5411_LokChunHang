"""
Configuration for the DIT5411 Traditional Chinese Handwriting Recognition project.
Edit these paths and hyperparameters before training.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Where the raw images are stored. The full dataset unzips to a 'cleaned_data'
# folder with subdirectories named by Chinese characters.
# Example structure:
#   data/raw/cleaned_data/一/一_0001.png
# If you use the 50x50 common-words dataset, the folder is 'cleaned_data(50_50)'.
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "cleaned_data")

# Alternative path for the 50x50 common words dataset
RAW_DATA_DIR_50x50 = os.path.join(BASE_DIR, "data", "raw", "cleaned_data(50_50)")

# Where processed train / test splits will be saved
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")

# Optional: where explicitly augmented PNGs are saved (slow, large disk usage)
AUGMENTED_DIR = os.path.join(BASE_DIR, "data", "augmented")

# Model checkpoints and results
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ---------------------------------------------------------------------------
# Data settings
# ---------------------------------------------------------------------------
# Number of samples per class to keep for training (assignment asks for 40)
TRAIN_SAMPLES_PER_CLASS = 40

# Number of augmented samples per training sample.
# 40 original * 5 augmented = 200 effective training samples per class.
AUGMENT_FACTOR = 5

# Target image size. The full dataset is 300x300 but resized for speed.
# The 50x50 dataset is also resized to this size.
IMG_SIZE = 48

# Grayscale input channel
IMG_CHANNELS = 1

# Number of classes. The assignment specifies 13,065 characters.
# If some classes have fewer samples than expected, the script will use only
# classes that actually exist in the dataset.
NUM_CLASSES = 13065

# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------
BATCH_SIZE = 1024
EPOCHS = 5
LEARNING_RATE = 3e-4
DROPOUT_RATE = 0.3

# ---------------------------------------------------------------------------
# Random seed for reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Ensure directories exist
# ---------------------------------------------------------------------------
for d in [MODELS_DIR, RESULTS_DIR, TRAIN_DIR, TEST_DIR]:
    os.makedirs(d, exist_ok=True)
