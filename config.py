"""
Configuration file for Image Caption Generator
"""

import os

# Dataset paths
BASE_DIR = './data'
WORKING_DIR = './models'

# Model paths
MODEL_PATH = os.path.join(WORKING_DIR, 'best_model.h5')
MODEL_WEIGHTS_PATH = os.path.join(WORKING_DIR, 'model_weights.h5')
MAX_LENGTH_PATH = os.path.join(WORKING_DIR, 'max_length.pkl')
FEATURES_PATH = os.path.join(WORKING_DIR, 'features.pkl')
TOKENIZER_PATH = os.path.join(WORKING_DIR, 'tokenizer.pkl')
MAPPING_PATH = os.path.join(WORKING_DIR, 'mapping.pkl')

# Image paths
IMAGES_DIR = os.path.join(BASE_DIR, 'Images')
CAPTIONS_FILE = os.path.join(BASE_DIR, 'captions.txt')

# Model parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
EMBEDDING_DIM = 256
LSTM_UNITS = 256
DROPOUT_RATE = 0.4

# Create directories if they don't exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)
