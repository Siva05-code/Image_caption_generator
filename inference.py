"""
Inference script for Image Caption Generator
Quick testing without UI
"""

import os
import sys
from PIL import Image
import numpy as np
from model import build_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

from config import (
    MODEL_WEIGHTS_PATH, TOKENIZER_PATH, IMG_SIZE
)
from utils import (
    load_tokenizer, create_vgg_model, predict_caption, load_max_length
)


def load_image(image_path):
    """Load and preprocess image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    image_array = img_to_array(image)
    image_array = image_array.reshape((1, IMG_SIZE, IMG_SIZE, 3))
    image_array = image_array.astype('float32')
    image_array = preprocess_input(image_array)
    
    return image_array, Image.open(image_path)


def generate_caption_for_image(image_path, max_length=None):
    """Generate caption for a single image"""
    
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    
    # Load max_length from saved file, or use provided value
    if max_length is None:
        max_length = load_max_length()
    
    print("Building and loading model...")
    vocab_size = len(tokenizer.word_index) + 1
    model = build_model(vocab_size, max_length)
    model.load_weights(MODEL_WEIGHTS_PATH)
    
    print("Loading VGG16 model...")
    vgg_model = create_vgg_model()
    
    print(f"\nGenerating caption for: {image_path}")
    
    # Load and preprocess image
    image_array, pil_image = load_image(image_path)
    
    # Extract features
    features = vgg_model.predict(image_array, verbose=0)
    
    # Generate caption
    caption = predict_caption(model, features, tokenizer, max_length)
    
    # Clean up caption
    caption = caption.replace('startseq', '').replace('endseq', '').strip()
    caption = caption.capitalize()
    
    return caption, pil_image


def main():
    """Main inference function"""
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        print("Example: python inference.py image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        caption, image = generate_caption_for_image(image_path)
        
        print("\n" + "=" * 50)
        print("GENERATED CAPTION:")
        print("=" * 50)
        print(caption)
        print("=" * 50)
        
        # Optionally display image
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            plt.imshow(image)
            plt.axis('off')
            plt.title(caption, fontsize=12, wrap=True)
            plt.tight_layout()
            plt.savefig('caption_result.png', dpi=100, bbox_inches='tight')
            print("\nImage saved as 'caption_result.png'")
            plt.show()
        except ImportError:
            print("\nMatplotlib not available. Skipping image display.")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
