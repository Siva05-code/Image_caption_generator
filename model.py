"""
Model architecture for Image Caption Generator
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Embedding, Dropout, add
)
from tensorflow.keras.utils import plot_model
from config import EMBEDDING_DIM, LSTM_UNITS, DROPOUT_RATE


def build_model(vocab_size, max_length):
    """
    Build the image caption generator model
    
    Args:
        vocab_size: Size of the vocabulary
        max_length: Maximum length of captions
    
    Returns:
        model: Compiled Keras model
    """
    
    # Encoder - Image feature layers
    inputs1 = Input(shape=(4096,), name='image_input')
    fe1 = Dropout(DROPOUT_RATE)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # Sequence feature layers
    inputs2 = Input(shape=(max_length,), name='caption_input')
    se1 = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True)(inputs2)
    se2 = Dropout(DROPOUT_RATE)(se1)
    se3 = LSTM(LSTM_UNITS)(se2)
    
    # Decoder - Merge and decode
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # Create model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model


def plot_model_architecture(model, filepath='model_architecture.png'):
    """Plot and save model architecture (optional - requires graphviz)"""
    try:
        plot_model(model, to_file=filepath, show_shapes=True)
        print(f"Model architecture saved to {filepath}")
    except ImportError:
        print("⚠️  Graphviz not installed - skipping model visualization")
        print("   (This is optional and doesn't affect training)")
