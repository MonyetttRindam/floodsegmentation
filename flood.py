import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from huggingface_hub import hf_hub_download
import io
import sys
import streamlit as st

st.write("Python version:", sys.version)

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Flood Area Segmentation",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ==================== CUSTOM METRICS ====================
def dice_coefficient(y_true, y_pred, smooth=1):
    """Dice coefficient metric"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + 
                                           tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss function"""
    return 1 - dice_coefficient(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1):
    """Intersection over Union (IoU) metric"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


# ==================== MODEL LOADING ====================
@st.cache_resource
def load_resnet_unet():
    """Load ResNet-UNet model from Hugging Face"""
    try:
        with st.spinner("Loading ResNet-UNet model..."):
            model_path = hf_hub_download(
                repo_id="MonyetttRindam/floodunetresnet", 
                filename="resnet_unet_best.h5"
            )
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'dice_coefficient': dice_coefficient,
                    'dice_loss': dice_loss,
                    'iou_metric': iou_metric
                }
            )
            return model
    except Exception as e:
        st.error(f"Error loading ResNet-UNet model: {e}")
        return None

@st.cache_resource
def load_unet():
    """Load U-Net model from Hugging Face"""
    try:
        with st.spinner("Loading U-Net model..."):
            model_path = hf_hub_download(
                repo_id="MonyetttRindam/floodunet", 
                filename="unet_best.h5"
            )
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'dice_coefficient': dice_coefficient,
                    'dice_loss': dice_loss,
                    'iou_metric': iou_metric
                }
            )
            return model
    except Exception as e:
        st.error(f"Error loading U-Net model: {e}")
        return None


# ==================== IMAGE PROCESSING ====================
def preprocess_image(image, target_size=(512, 512)):
    """Preprocess image for model prediction"""
    # Resize
    img_resized = cv2.resize(np.array(image), target_size)
    
    # Normalize
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img_resized

def post_process_mask(mask, threshold=0.5, min_area=100):
    """Post-process predicted mask"""
    # Binarize
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            binary_mask[labels == i] = 0
    
    return binary_mask

def create_overlay(image, mask, alpha=0.4):
    """Create overlay of mask on original image"""
    overlay = image.copy()
    overlay[mask > 0] = [0, 0, 255]  # Blue for flood area
    blended = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
    return blended


# ==================== PREDICTION ====================
def predict_flood(image, model, model_name):
    """Predict flood area using selected model"""
    # Preprocess
    img_batch, img_resized = preprocess_image(image)
    
    # Predict
    with st.spinner(f"Predicting with {model_name}..."):
        prediction = model.predict(img_batch, verbose=0)[0]
    
    # Extract mask
    mask_raw = prediction[:, :, 0]
    mask_processed = post_process_mask(mask_raw)
    
    # Calculate flood percentage
    flood_pixels = np.sum(mask_processed)
    total_pixels = mask_processed.size
    flood_percentage = (flood_pixels / total_pixels) * 100
    
    return {
        'original': img_resized,
        'mask_raw': mask_raw,
        'mask_processed': mask_processed,
        'flood_percentage': flood_percentage,
        'overlay': create_overlay(img_resized, mask_processed)
    }


# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<div class="main-header">🌊 Flood Area Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deteksi Otomatis Area Banjir menggunakan Deep Learning</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Pilih Model",
        ["ResNet-UNet (Recommended)", "U-Net", "Both (Comparison)"]
    )
    
    # Threshold slider
    threshold = st.sidebar.slider(
        "Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Threshold untuk binarisasi mask"
    )
    
    # Min area slider
    min_area = st.sidebar.slider(
        "Minimum Area (pixels)",
        min_value=50,
        max_value=500,
        value=100,
        step=50,
        help="Area minimum untuk mempertahankan komponen"
    )
    
    st.sidebar.markdown("---")
    
    # Model info
    st.sidebar.header("📊 Model Information")
    st.sidebar.markdown("""
    **U-Net**
    - Parameters: ~31M
    - Dice Score: 0.859
    - IoU: 0.761
    
    **ResNet-UNet**
    - Parameters: ~35M
    - Dice Score: 0.884 ⭐
    - IoU: 0.797 ⭐
    """)
    
    # Load models based on selection
    resnet_model = None
    unet_model = None
    
    if "ResNet" in model_choice:
        resnet_model = load_resnet_unet()
        if resnet_model is None:
            st.error("Failed to load ResNet-UNet model. Please refresh the page.")
            return
    
    if "U-Net" in model_choice or "Both" in model_choice:
        unet_model = load_unet()
        if unet_model is None:
            st.error("Failed to load U-Net model. Please refresh the page.")
            return
    
    # File uploader
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image of flood area...",
        type=["jpg", "jpeg", "png"],
        help="Upload gambar satelit atau aerial dari area banjir"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🖼️ Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("### ℹ️ Image Info")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Format:** {image.format}")
        
        # Predict button
        if st.button("🚀 Predict Flood Area", type="primary"):
            
            if "Both" in model_choice:
                # Predict with both models
                st.markdown("---")
                st.markdown("### 🔍 Prediction Results - Comparison")
                
                col1, col2 = st.columns(2)
                
                # U-Net prediction
                with col1:
                    st.markdown("#### U-Net")
                    results_unet = predict_flood(image, unet_model, "U-Net")
                    
                    # Display metrics
                    st.metric("Flood Coverage", f"{results_unet['flood_percentage']:.2f}%")
                    
                    # Display results
                    tab1, tab2, tab3 = st.tabs(["Raw Prediction", "Processed Mask", "Overlay"])
                    
                    with tab1:
                        st.image(results_unet['mask_raw'], use_container_width=True, clamp=True)
                    with tab2:
                        st.image(results_unet['mask_processed'], use_container_width=True, clamp=True)
                    with tab3:
                        st.image(results_unet['overlay'], use_container_width=True, channels="RGB")
                
                # ResNet-UNet prediction
                with col2:
                    st.markdown("#### ResNet-UNet")
                    results_resnet = predict_flood(image, resnet_model, "ResNet-UNet")
                    
                    # Display metrics
                    st.metric("Flood Coverage", f"{results_resnet['flood_percentage']:.2f}%")
                    
                    # Display results
                    tab1, tab2, tab3 = st.tabs(["Raw Prediction", "Processed Mask", "Overlay"])
                    
                    with tab1:
                        st.image(results_resnet['mask_raw'], use_container_width=True, clamp=True)
                    with tab2:
                        st.image(results_resnet['mask_processed'], use_container_width=True, clamp=True)
                    with tab3:
                        st.image(results_resnet['overlay'], use_container_width=True, channels="RGB")
                
                # Comparison
                st.markdown("#### 📊 Comparison")
                diff = abs(results_unet['flood_percentage'] - results_resnet['flood_percentage'])
                st.info(f"Difference in flood coverage: {diff:.2f}%")
                
            else:
                # Single model prediction
                model = resnet_model if "ResNet" in model_choice else unet_model
                model_name = "ResNet-UNet" if "ResNet" in model_choice else "U-Net"
                
                results = predict_flood(image, model, model_name)
                
                st.markdown("---")
                st.markdown(f"### 🔍 Prediction Results - {model_name}")
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model", model_name)
                with col2:
                    st.metric("Flood Coverage", f"{results['flood_percentage']:.2f}%")
                with col3:
                    flood_area = (results['flood_percentage'] / 100) * (512 * 512)
                    st.metric("Flood Pixels", f"{int(flood_area):,}")
                
                # Display results
                st.markdown("#### Visualization")
                tab1, tab2, tab3, tab4 = st.tabs(["Original", "Raw Prediction", "Processed Mask", "Overlay"])
                
                with tab1:
                    st.image(results['original'], use_container_width=True, channels="RGB")
                with tab2:
                    st.image(results['mask_raw'], use_container_width=True, clamp=True)
                with tab3:
                    st.image(results['mask_processed'], use_container_width=True, clamp=True)
                with tab4:
                    st.image(results['overlay'], use_container_width=True, channels="RGB")
                
                # Download button
                st.markdown("#### 💾 Download Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Save processed mask
                    mask_pil = Image.fromarray((results['mask_processed'] * 255).astype(np.uint8))
                    buf = io.BytesIO()
                    mask_pil.save(buf, format='PNG')
                    st.download_button(
                        label="Download Mask",
                        data=buf.getvalue(),
                        file_name="flood_mask.png",
                        mime="image/png"
                    )
                
                with col2:
                    # Save overlay
                    overlay_pil = Image.fromarray(results['overlay'])
                    buf = io.BytesIO()
                    overlay_pil.save(buf, format='PNG')
                    st.download_button(
                        label="Download Overlay",
                        data=buf.getvalue(),
                        file_name="flood_overlay.png",
                        mime="image/png"
                    )
    
    else:
        # Instructions
        st.info("👆 Please upload an image to get started")
        
        # Example images section
        st.markdown("---")
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. **Upload** an aerial or satellite image of a flood-affected area
        2. **Select** a model (ResNet-UNet recommended for best accuracy)
        3. **Adjust** threshold and minimum area if needed
        4. **Click** 'Predict Flood Area' button
        5. **View** and **download** the results
        """)
        
        st.markdown("### ℹ️ About")
        st.markdown("""
        This application uses deep learning models to automatically detect and segment flood-affected areas in satellite/aerial imagery.
        
        **Models:**
        - **U-Net**: Classic encoder-decoder architecture
        - **ResNet-UNet**: U-Net with ResNet50 encoder (transfer learning)
        
        **Use Cases:**
        - Disaster response and assessment
        - Flood extent mapping
        - Historical flood analysis
        - Risk assessment
        """)


# ==================== FOOTER ====================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌊 Flood Area Segmentation | Built with Streamlit & TensorFlow</p>
        <p>Models hosted on 🤗 Hugging Face</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":

    main()
