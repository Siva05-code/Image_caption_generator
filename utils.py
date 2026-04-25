"""
Utility functions for Image Caption Generator
"""

import os
import pickle
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from config import (
    BASE_DIR, WORKING_DIR, IMAGES_DIR, CAPTIONS_FILE,
    FEATURES_PATH, TOKENIZER_PATH, MAPPING_PATH, IMG_SIZE
)
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def create_vgg_model():
    """Create VGG16 model for feature extraction"""
    model = VGG16()
    model = model.__class__(inputs=model.inputs, outputs=model.layers[-2].output)
    return model


def extract_features(vgg_model, images_dir=IMAGES_DIR):
    """Extract features from all images using VGG16"""
    features = {}
    
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_name in tqdm(image_files, desc="Extracting features"):
        img_path = os.path.join(images_dir, img_name)
        
        # Load and preprocess image
        image = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        
        # Extract features
        feature = vgg_model.predict(image, verbose=0)
        image_id = img_name.split('.')[0]
        features[image_id] = feature
    
    return features


def save_features(features, path=FEATURES_PATH):
    """Save extracted features to pickle file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(features, f)
    print(f"Features saved to {path}")


def load_features(path=FEATURES_PATH):
    """Load features from pickle file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features file not found: {path}")
    with open(path, 'rb') as f:
        features = pickle.load(f)
    return features


def load_captions(captions_file=CAPTIONS_FILE):
    """Load captions from text file"""
    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"Captions file not found: {captions_file}")
    
    with open(captions_file, 'r') as f:
        next(f)  # Skip header
        captions_doc = f.read()
    
    return captions_doc


def create_caption_mapping(captions_doc):
    """Create mapping of image ID to captions"""
    mapping = {}
    
    for line in tqdm(captions_doc.split('\n'), desc="Processing captions"):
        tokens = line.split(',')
        if len(line) < 2:
            continue
        
        image_id = tokens[0].split('.')[0]
        caption = " ".join(tokens[1:])
        
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
    
    return mapping


def clean_captions(mapping):
    """Clean and preprocess captions"""
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            
            # Convert to lowercase
            caption = caption.lower()
            
            # Remove special characters and digits
            caption = re.sub(r'[^a-z\s]', '', caption)
            
            # Remove extra spaces
            caption = re.sub(r'\s+', ' ', caption).strip()
            
            # Add start and end tags
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            captions[i] = caption


def save_tokenizer(tokenizer, path=TOKENIZER_PATH):
    """Save tokenizer to pickle file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {path}")


def load_tokenizer(path=TOKENIZER_PATH):
    """Load tokenizer from pickle file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tokenizer file not found: {path}")
    with open(path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


def create_tokenizer(all_captions):
    """Create tokenizer from all captions"""
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    return tokenizer


def save_mapping(mapping, path=MAPPING_PATH):
    """Save caption mapping to pickle file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(mapping, f)
    print(f"Mapping saved to {path}")


def load_mapping(path=MAPPING_PATH):
    """Load caption mapping from pickle file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mapping file not found: {path}")
    with open(path, 'rb') as f:
        mapping = pickle.load(f)
    return mapping


def save_max_length(max_length, path=None):
    """Save max_length to pickle file"""
    from config import MAX_LENGTH_PATH
    if path is None:
        path = MAX_LENGTH_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(max_length, f)
    print(f"Max length ({max_length}) saved to {path}")


def load_max_length(path=None):
    """Load max_length from pickle file"""
    from config import MAX_LENGTH_PATH
    if path is None:
        path = MAX_LENGTH_PATH
    if not os.path.exists(path):
        # Return default if not found
        return 35
    with open(path, 'rb') as f:
        max_length = pickle.load(f)
    return max_length


def idx_to_word(integer, tokenizer):
    """Convert index back to word using tokenizer"""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image_features, tokenizer, max_length):
    """Generate caption for an image"""
    in_text = 'startseq'
    
    for i in range(max_length):
        # Encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Predict next word
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        
        # Convert index to word
        word = idx_to_word(yhat, tokenizer)
        
        if word is None:
            break
        
        in_text += " " + word
        
        if word == 'endseq':
            break
    
    return in_text


def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    """Generate batches of training data"""
    X1, X2, y = list(), list(), list()
    n = 0
    
    while True:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            
            for caption in captions:
                # Encode the sequence
                seq = tokenizer.texts_to_sequences([caption])[0]
                
                # Split into X, y pairs
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq_categorical = np.zeros(vocab_size)
                    out_seq_categorical[out_seq] = 1
                    
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq_categorical)
            
            if n == batch_size:
                # Convert to numpy arrays
                X1_batch = np.array(X1, dtype=np.float32)
                X2_batch = np.array(X2, dtype=np.int32)
                y_batch = np.array(y, dtype=np.float32)
                
                yield (X1_batch, X2_batch), y_batch
                X1, X2, y = list(), list(), list()
                n = 0
