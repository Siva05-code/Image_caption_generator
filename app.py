"""
Streamlit UI for Image Caption Generator
"""

import streamlit as st
import os
from PIL import Image
import numpy as np
from model import build_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

from config import (
    MODEL_WEIGHTS_PATH, FEATURES_PATH, TOKENIZER_PATH, MAPPING_PATH, IMG_SIZE
)
from utils import (
    load_features, load_tokenizer, load_mapping, load_max_length,
    predict_caption, create_vgg_model
)


# Page configuration
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1f77b4;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model_resources():
    """Load model, tokenizer, and features with caching"""
    try:
        # Check if model exists
        if not os.path.exists(MODEL_WEIGHTS_PATH):
            st.error(f"❌ Model not found at {MODEL_WEIGHTS_PATH}")
            st.info("Please run training script first: `python train.py`")
            return None, None, None, None
        
        # Load model - rebuild from architecture and load weights
        with st.spinner("Loading model..."):
            try:
                # Load tokenizer first to get vocab_size
                tokenizer = load_tokenizer(TOKENIZER_PATH)
                vocab_size = len(tokenizer.word_index) + 1
                # Load the actual max_length used during training
                max_length = load_max_length()
                # Build model architecture and load weights
                model = build_model(vocab_size, max_length)
                model.load_weights(MODEL_WEIGHTS_PATH)
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return None, None, None, None
        
        # Load VGG model for feature extraction
        with st.spinner("Loading VGG16 model..."):
            vgg_model = create_vgg_model()
        
        # Load mapping if available
        try:
            mapping = load_mapping(MAPPING_PATH)
        except:
            mapping = {}
        
        return model, tokenizer, vgg_model, mapping
    
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None, None, None, None


def extract_image_features(image_array, vgg_model):
    """Extract features from a single image"""
    image = image_array.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image)
    
    if len(image_array.shape) == 2:  # Grayscale
        image_array = np.stack([image_array] * 3, axis=-1)
    
    image_array = image_array.reshape((1, IMG_SIZE, IMG_SIZE, 3))
    image_array = image_array.astype('float32')
    image_array = preprocess_input(image_array)
    
    features = vgg_model.predict(image_array, verbose=0)
    return features


def generate_caption(model, image_features, tokenizer, max_length=35):
    """Generate caption for image features"""
    return predict_caption(model, image_features, tokenizer, max_length)


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("# 🖼️ Image Caption Generator")
    st.markdown(
        "Generate descriptive captions for images using CNN-LSTM deep learning model",
        unsafe_allow_html=True
    )
    
    # Load resources
    model, tokenizer, vgg_model, mapping = load_model_resources()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Options")
        
        mode = st.radio(
            "Select Mode",
            ["Upload Image", "Test on Dataset"],
            help="Choose how to test the model"
        )
        
        max_length = st.slider(
            "Max Caption Length",
            min_value=10,
            max_value=50,
            value=35,
            step=5
        )
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.info(f"""
        - **Model**: CNN-LSTM
        - **Image Encoder**: VGG16
        - **Vocabulary Size**: {len(tokenizer.word_index) + 1}
        """)
    
    # Main content
    if mode == "Upload Image":
        st.markdown("## Upload an Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload an image to generate a caption"
        )
        
        if uploaded_file is not None:
            # Display image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, width='stretch', caption="Uploaded Image")
            
            with col2:
                st.markdown("### 📝 Generated Caption")
                
                # Extract features and generate caption
                if st.button("🔄 Generate Caption", key="gen_caption", width='stretch'):
                    with st.spinner("Generating caption..."):
                        try:
                            # Extract features
                            features = extract_image_features(image, vgg_model)
                            
                            # Generate caption
                            caption = generate_caption(model, features, tokenizer, max_length)
                            
                            # Clean up caption
                            caption = caption.replace('startseq', '').replace('endseq', '').strip()
                            caption = caption.capitalize()
                            
                            # Display caption
                            st.success("✅ Caption generated successfully!")
                            st.markdown(f"""
                            <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;'>
                                <h4>Generated Caption:</h4>
                                <p style='font-size: 1.1rem; color: #1f77b4;'>{caption}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error generating caption: {str(e)}")
    
    else:  # Test on Dataset
        st.markdown("## Test on Dataset")
        
        if not mapping:
            st.warning("⚠️ No dataset mapping found. Please train the model first.")
        else:
            # Select image from dataset
            image_ids = list(mapping.keys())[:10]  # Limit to first 10 for demo
            
            if image_ids:
                selected_image_id = st.selectbox(
                    "Select an image from the test set",
                    image_ids,
                    help="Choose an image to test"
                )
                
                if st.button("🔄 Generate Caption", key="gen_dataset_caption", width='stretch'):
                    # Try to load image from dataset
                    try:
                        features = load_features()
                        
                        if selected_image_id in features:
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.markdown("### Original Captions")
                                captions = mapping.get(selected_image_id, [])
                                for i, caption in enumerate(captions, 1):
                                    caption_clean = caption.replace('startseq', '').replace('endseq', '').strip()
                                    st.write(f"**{i}.** {caption_clean}")
                            
                            with col2:
                                st.markdown("### 📝 Generated Caption")
                                
                                with st.spinner("Generating caption..."):
                                    # Generate caption
                                    caption = generate_caption(
                                        model,
                                        features[selected_image_id],
                                        tokenizer,
                                        max_length
                                    )
                                    
                                    # Clean up caption
                                    caption = caption.replace('startseq', '').replace('endseq', '').strip()
                                    caption = caption.capitalize()
                                    
                                    st.success("✅ Caption generated successfully!")
                                    st.markdown(f"""
                                    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;'>
                                        <p style='font-size: 1.1rem; color: #1f77b4;'>{caption}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.error(f"Features not found for image {selected_image_id}")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.info("No images available in the dataset.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "Image Caption Generator | CNN-LSTM Model | Built with Streamlit"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
