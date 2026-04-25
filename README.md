# 🖼️ Image Caption Generator using CNN-LSTM

A production-ready deep learning project that generates descriptive captions for images using Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset Setup](#dataset-setup)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Inference](#inference)
- [Using the Web UI](#using-the-web-ui)
- [Results & Evaluation](#results--evaluation)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## 📌 Overview

This project implements an **end-to-end image captioning system** that understands visual content and generates natural language descriptions. It bridges computer vision and NLP by combining:
- **VGG16** pre-trained CNN for feature extraction
- **LSTM** networks for sequential caption generation
- **Flickr8k dataset** (8,000 images with 5 captions each)

**Use Cases:**
- Assistive technology for visually impaired users
- Content-based image retrieval
- Social media image auto-tagging
- Accessibility enhancement

---

## ✨ Features

✅ Pre-trained VGG16 for robust image feature extraction  
✅ LSTM-based sequence generation with attention-like mechanism  
✅ Trained on diverse Flickr8k dataset (8,000 images, 40,000 captions)  
✅ BLEU score evaluation for caption quality assessment  
✅ Interactive Streamlit web UI for real-time testing  
✅ Command-line inference script for batch processing  
✅ Modular, production-ready code structure  
✅ Comprehensive configuration management  

---

## 📦 Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- 4GB+ RAM (8GB+ recommended for training)
- GPU support optional but recommended for training (NVIDIA GPU with CUDA)

---

## 🚀 Installation

### Step 1: Clone/Navigate to Project Directory

```bash
cd /path/to/DL_prj
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using the provided Makefile
make setup
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt

# Download NLTK data for BLEU score calculation
python3 -c "import nltk; nltk.download('punkt')"

# OR using Makefile
make install
```

### Required Packages

- TensorFlow >= 2.14.0
- NumPy >= 1.24.0
- OpenCV, Pillow (image processing)
- scikit-learn (metrics)
- NLTK (NLP utilities)
- Streamlit (web UI)

---

## 📂 Project Structure

```
DL_prj/
├── app.py                          # Streamlit web application
├── train.py                        # Model training script
├── inference.py                    # Standalone inference script
├── model.py                        # Model architecture definition
├── utils.py                        # Utility functions (data loading, preprocessing)
├── config.py                       # Configuration parameters
├── requirements.txt                # Project dependencies
├── Makefile                        # Convenient commands
├── README.md                       # This file
│
├── data/
│   ├── captions.txt               # Caption annotations
│   └── Images/                    # Image dataset (download required)
│
└── models/
    ├── best_model.h5             # Trained model (weights + architecture)
    ├── model_weights.h5          # Extracted weights only
    ├── tokenizer.pkl             # Word tokenizer
    ├── mapping.pkl               # Image-caption mapping
    ├── features.pkl              # Extracted image features
    └── max_length.pkl            # Maximum caption length
```

---

## 📊 Dataset Setup

### Download Flickr8k Dataset

1. **Download from Kaggle:**
   ```bash
   # Requires Kaggle API
   kaggle datasets download -d adityajn105/flickr8k
   unzip flickr8k.zip
   ```

2. **Or download manually:**
   - Visit: [Flickr8k on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
   - Download and extract to your local machine

### Organize Dataset

```bash
# Expected directory structure:
data/
├── Images/
│   ├── 1000268201_693b08cb0e.jpg
│   ├── 1001773457_577c3a7d70.jpg
│   └── ... (8,000 images total)
└── captions.txt
```

### Caption File Format

`captions.txt` should contain comma-separated image names and captions:

```
image_name,caption1|caption2|caption3|caption4|caption5
1000268201_693b08cb0e.jpg,A man with a net is catching a butterfly|a person holding a net in front of a butterfly|...
1001773457_577c3a7d70.jpg,A girl going into a wooden building|a young girl in a checkered jumper jumping into a wooden structure|...
```

---

## ⚡ Quick Start

### Option 1: Using Makefile (Recommended)

```bash
# Show all available commands
make help

# Install dependencies
make install

# Train the model
make train

# Run the Streamlit UI
make run-ui

# Run inference on an image
IMAGE=path/to/image.jpg make inference

# Clean temporary files
make clean
```

### Option 2: Using Python Directly

```bash
# Train model
python train.py

# Run web UI
streamlit run app.py

# Run inference
python inference.py path/to/image.jpg
```

---

## 🏗️ Model Architecture

### System Overview

```
Image Input (224×224)
        ↓
    [VGG16 Encoder]
        ↓
  Feature Vector (4096-dim)
        ↓
    [Dense Layer]
        ↓
    ┌─────┬─────┐
    ↓     ↓     ↓
  Caption ← [LSTM Decoder]
  Input   └─────┬─────┘
    ↓           ↓
[Embedding]   [Dense]
    ↓           ↓
  [LSTM]        ↓
    └─────┬─────┘
          ↓
    [Dense + Softmax]
          ↓
      Word Output
```

### Component Details

#### **Encoder (Feature Extraction)**
- **Architecture:** VGG16 pre-trained on ImageNet
- **Input:** Images (224×224×3)
- **Output:** Feature vector (4096-dimensional)
- **Purpose:** Extract high-level visual features from images

#### **Decoder (Caption Generation)**
- **Components:**
  - Embedding layer (converts word indices to dense vectors)
  - LSTM layer with dropout (processes sequence sequentially)
  - Dense output layer with softmax (predicts next word)
  
- **Parameters:**
  - Embedding dimension: 256
  - LSTM units: 256
  - Dropout rate: 0.4
  - Output vocabulary: ~7,000 unique words

### Training Strategy

**Teacher Forcing:** During training, the model receives the ground-truth previous word as input, accelerating convergence.

**Sequence-to-Sequence:** 
- Input: Image features + previous caption word
- Output: Probability distribution over vocabulary for next word

---

## 📈 Training

### Training Process

```bash
python train.py
```

#### What Happens:

1. **Data Preparation** (5 min)
   - Load captions from `captions.txt`
   - Create image-to-caption mapping
   - Clean and tokenize captions
   - Build vocabulary

2. **Feature Extraction** (15-30 min)
   - Load pre-trained VGG16
   - Extract features for all 8,000 images
   - Save features to `features.pkl`

3. **Model Training** (varies by hardware)
   - Build CNN-LSTM model
   - Train for 20 epochs with batch size 32
   - Monitor BLEU score on validation set
   - Save best model to `best_model.h5`

### Training Configuration

Edit `config.py` to customize:

```python
BATCH_SIZE = 32        # Batch size for training
EPOCHS = 20            # Number of epochs
EMBEDDING_DIM = 256    # Word embedding dimension
LSTM_UNITS = 256       # LSTM hidden units
DROPOUT_RATE = 0.4     # Dropout for regularization
```

### Expected Results

- **Training Time:** 2-6 hours (CPU), 30-60 minutes (GPU)
- **BLEU Score:** 0.45-0.55 (typical)
- **Model Size:** ~100-150 MB

---

## 🔍 Inference

### Method 1: Python Script

```bash
python inference.py path/to/image.jpg
```

**Output:**
```
Loading model...
Loading tokenizer...
Generating caption...

Generated Caption:
"A man in a blue shirt is playing a guitar"
```

### Method 2: Interactive Web UI (Recommended)

```bash
streamlit run app.py
```

Then visit: `http://localhost:8501`

**Features:**
- Upload images directly
- View generated captions in real-time
- Visualize image with caption overlay
- Try multiple images without reloading

### Method 3: Python Code

```python
from inference import generate_caption_for_image

caption = generate_caption_for_image("path/to/image.jpg")
print(f"Caption: {caption}")
```

---

## 🌐 Using the Web UI

### Streamlit Application Features

1. **Upload Image**
   - Drag and drop or click to browse
   - Supports JPG, PNG, JPEG formats

2. **Real-time Generation**
   - Automatic caption generation on upload
   - Display processed image with caption

3. **Results Display**
   - Original image with caption overlay
   - Generation time metrics
   - Confidence scores (if available)

### Running the UI

```bash
# Basic run
streamlit run app.py

# Custom port
streamlit run app.py --server.port 8502

# Development mode (hot-reload)
streamlit run app.py --logger.level=debug
```

---

## 📊 Results & Evaluation

### Performance Metrics

#### BLEU Score
- Compares generated captions with reference captions
- Range: 0-1 (higher is better)
- BLEU-1 through BLEU-4 (1-gram to 4-gram precision)

**Expected Scores:**
- BLEU-1: 0.65-0.75 (word-level precision)
- BLEU-4: 0.45-0.55 (sentence-level precision)

### Evaluation Script

```python
from nltk.translate.bleu_score import corpus_bleu

# Calculate BLEU score on test set
bleu_score = corpus_bleu(reference_captions, generated_captions)
print(f"BLEU Score: {bleu_score:.4f}")
```

### Sample Outputs

```
Image 1: "A dog running in the park"
Image 2: "People sitting at a dinner table"
Image 3: "A person riding a bicycle on a road"
```

---

## 🛠️ Configuration

### Model Hyperparameters (`config.py`)

```python
# Image processing
IMG_SIZE = 224                  # Input image size

# Training
BATCH_SIZE = 32                 # Batch size
EPOCHS = 20                     # Training epochs

# Model architecture
EMBEDDING_DIM = 256             # Word embedding dimension
LSTM_UNITS = 256                # LSTM hidden units
DROPOUT_RATE = 0.4              # Dropout rate for regularization

# Data paths
BASE_DIR = './data'             # Data directory
WORKING_DIR = './models'        # Models directory
```

---

## 🔧 Troubleshooting

### Common Issues

#### Issue: "No module named 'tensorflow'"
```bash
pip install tensorflow --upgrade
```

#### Issue: "NLTK data not found"
```bash
python3 -c "import nltk; nltk.download('punkt')"
```

#### Issue: Out of Memory during training
```python
# In config.py, reduce batch size
BATCH_SIZE = 16  # Instead of 32

# Or reduce image processing
LSTM_UNITS = 128  # Instead of 256
```

#### Issue: Model file not found
```bash
# Ensure best_model.h5 exists in models/
ls -lh models/best_model.h5

# If missing, retrain
python train.py
```

#### Issue: Image format not supported
```bash
# Convert image format first
from PIL import Image
img = Image.open('image.png').convert('RGB')
img.save('image.jpg')
```

---

## 📚 References

- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
- [LSTM Networks](https://en.wikipedia.org/wiki/Long_short-term_memory)
- [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [BLEU Score Evaluation](https://en.wikipedia.org/wiki/BLEU)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)

---

## 📄 License

This project is provided as-is for educational purposes.

---

## 🤝 Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the configuration in `config.py`
3. Ensure dataset is properly formatted in `data/`
4. Check that all dependencies are installed with `pip list`

---

**Last Updated:** April 2026  
**Python Version:** 3.8+  
**TensorFlow Version:** 2.14.0+

