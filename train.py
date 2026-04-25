"""
Training script for Image Caption Generator
"""

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

from config import (
    BASE_DIR, WORKING_DIR, BATCH_SIZE, EPOCHS,
    MODEL_PATH, MODEL_WEIGHTS_PATH, MAX_LENGTH_PATH, FEATURES_PATH, TOKENIZER_PATH, MAPPING_PATH
)
from model import build_model, plot_model_architecture
from utils import (
    create_vgg_model, extract_features, save_features, load_features,
    load_captions, create_caption_mapping, clean_captions,
    create_tokenizer, save_tokenizer, save_mapping, save_max_length,
    data_generator, predict_caption, idx_to_word
)


def prepare_data():
    """Prepare training data"""
    print("=" * 50)
    print("PREPARING DATA")
    print("=" * 50)
    
    # Load captions
    print("\n1. Loading captions...")
    captions_doc = load_captions()
    
    # Create mapping
    print("2. Creating caption mapping...")
    mapping = create_caption_mapping(captions_doc)
    print(f"   Total images: {len(mapping)}")
    
    # Clean captions
    print("3. Cleaning captions...")
    clean_captions(mapping)
    
    # Save mapping
    save_mapping(mapping)
    
    # Create all captions list
    all_captions = []
    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)
    
    print(f"   Total captions: {len(all_captions)}")
    
    # Create and save tokenizer
    print("4. Creating tokenizer...")
    tokenizer = create_tokenizer(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    save_tokenizer(tokenizer)
    print(f"   Vocabulary size: {vocab_size}")
    
    # Get max length
    max_length = max(len(caption.split()) for caption in all_captions)
    print(f"   Max caption length: {max_length}")
    
    return mapping, tokenizer, vocab_size, max_length, all_captions


def prepare_features():
    """Extract and prepare image features"""
    print("\n" + "=" * 50)
    print("EXTRACTING IMAGE FEATURES")
    print("=" * 50)
    
    # Check if features already exist
    if os.path.exists(FEATURES_PATH):
        print("Loading pre-extracted features...")
        features = load_features()
        print(f"Loaded features for {len(features)} images")
    else:
        print("Creating VGG16 model...")
        vgg_model = create_vgg_model()
        
        print("Extracting features from images...")
        features = extract_features(vgg_model)
        
        print(f"Extracted features for {len(features)} images")
        save_features(features)
    
    return features


def train_model(mapping, features, tokenizer, vocab_size, max_length):
    """Train the caption generator model"""
    print("\n" + "=" * 50)
    print("TRAINING MODEL")
    print("=" * 50)
    
    # Build model
    print("\nBuilding model...")
    model = build_model(vocab_size, max_length)
    plot_model_architecture(model)
    print(model.summary())
    
    # Train-test split
    image_ids = list(mapping.keys())
    split = int(len(image_ids) * 0.90)
    train_ids = image_ids[:split]
    
    print(f"\nTraining on {len(train_ids)} images")
    
    # Training loop
    steps_per_epoch = len(train_ids) // BATCH_SIZE
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        # Wrap generator with tf.data.Dataset
        dataset = tf.data.Dataset.from_generator(
            lambda: data_generator(
                train_ids, mapping, features, tokenizer,
                max_length, vocab_size, BATCH_SIZE
            ),
            output_signature=(
                (tf.TensorSpec(shape=(None, 4096), dtype=tf.float32),
                 tf.TensorSpec(shape=(None, max_length), dtype=tf.int32)),
                tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32)
            )
        )
        
        model.fit(dataset, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)
    
    # Save model weights
    os.makedirs(WORKING_DIR, exist_ok=True)
    model.save_weights(MODEL_WEIGHTS_PATH)
    print(f"\nModel weights saved to {MODEL_WEIGHTS_PATH}")
    
    # Save max_length for later use
    save_max_length(max_length)
    
    return model, train_ids


def evaluate_model(model, mapping, features, tokenizer, max_length, test_ids):
    """Evaluate model using BLEU score"""
    print("\n" + "=" * 50)
    print("EVALUATING MODEL")
    print("=" * 50)
    
    actual, predicted = list(), list()
    
    print("\nGenerating captions for test set...")
    for key in tqdm(test_ids):
        captions = mapping[key]
        
        # Predict caption
        y_pred = predict_caption(model, features[key], tokenizer, max_length)
        
        # Split into words
        actual_captions = [caption.split() for caption in captions]
        y_pred = y_pred.split()
        
        actual.append(actual_captions)
        predicted.append(y_pred)
    
    # Calculate BLEU scores
    print("\nBLEU Scores:")
    bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    
    return {
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'BLEU-3': bleu3,
        'BLEU-4': bleu4
    }


def main():
    """Main training pipeline"""
    try:
        # Prepare data
        mapping, tokenizer, vocab_size, max_length, all_captions = prepare_data()
        
        # Extract features
        features = prepare_features()
        
        # Train model
        model, train_ids = train_model(mapping, features, tokenizer, vocab_size, max_length)
        
        # Evaluate model
        image_ids = list(mapping.keys())
        split = int(len(image_ids) * 0.90)
        test_ids = image_ids[split:]
        
        if len(test_ids) > 0:
            evaluate_model(model, mapping, features, tokenizer, max_length, test_ids)
        
        print("\n" + "=" * 50)
        print("TRAINING COMPLETE!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
