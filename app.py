import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown
import os

# ============= CUSTOM CSS =============
def load_custom_css():
    st.markdown("""
    <style>
    /* Globalne postavke */
    .stApp {
        background-color: #0a0a0a;
        color: #ffffff;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #0a0a0a 0%, #1a1a1a 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #00d9ff;
        margin-bottom: 30px;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.1);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    
    .main-subtitle {
        font-size: 1rem;
        color: #00d9ff;
        letter-spacing: 2px;
        margin-top: 5px;
    }
    
    .system-status {
        position: absolute;
        top: 20px;
        right: 20px;
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid #00ff88;
        border-radius: 20px;
        padding: 8px 16px;
        font-size: 0.9rem;
        color: #00ff88;
    }
    
    .system-status::before {
        content: "●";
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Upload zone */
    .upload-zone {
        background: #1a1a1a;
        border: 2px dashed #00d9ff;
        border-radius: 15px;
        padding: 60px 20px;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-zone:hover {
        background: #252525;
        border-color: #00ffff;
        box-shadow: 0 0 30px rgba(0, 217, 255, 0.2);
    }
    
    /* Cards */
    .info-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%);
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .preprocessing-card {
        background: #1a1a1a;
        border: 1px solid #00d9ff;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        transition: transform 0.2s;
    }
    
    .preprocessing-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 217, 255, 0.3);
    }
    
    .step-number {
        background: #00d9ff;
        color: #0a0a0a;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    /* Results */
    .result-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%);
        border: 2px solid #00ff88;
        border-radius: 15px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 0 40px rgba(0, 255, 136, 0.2);
    }
    
    .diagnosis-text {
        font-size: 2rem;
        font-weight: 700;
        color: #00d9ff;
        text-align: center;
        margin: 20px 0;
        text-shadow: 0 0 10px rgba(0, 217, 255, 0.5);
    }
    
    .confidence-label {
        color: #b8b8b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00d9ff 0%, #00a8cc 100%);
        color: #000000;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-size: 0.85rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 217, 255, 0.3);
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #00ffff 0%, #00d9ff 100%);
        box-shadow: 0 4px 12px rgba(0, 255, 255, 0.5);
        transform: translateY(-1px);
    }

    /* Main analysis button - larger */
    .stButton > button[kind="primary"] {
        padding: 12px 30px;
        font-size: 0.95rem;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00d9ff 0%, #00ff88 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f0f0f;
        border-right: 1px solid #00d9ff;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #00d9ff;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ============= PREPROCESSING FUNKCIJE (tvoje) =============
def sharpen_image(img):
    """Sharpen sliku koristeci Gaussian blur"""
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return sharpened, blurred

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Primijeni CLAHE enhancement"""
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_img = clahe.apply(img_gray)
    clahe_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
    
    return clahe_img, clahe_rgb

def preprocess_image(img):
    """Kompletan preprocessing pipeline"""
    sharpened, blurred = sharpen_image(img)

    if len(sharpened.shape) == 3:
        gray = cv2.cvtColor(sharpened, cv2.COLOR_RGB2GRAY)
    else:
        gray = sharpened

    clahe_img, clahe_rgb = apply_clahe(gray)

    # Resize na 224x224 za model
    clahe_rgb_resized = cv2.resize(clahe_rgb, (224, 224), interpolation=cv2.INTER_AREA)

    intermediate_steps = {
        'original': img,
        'blurred': blurred,
        'sharpened': sharpened,
        'grayscale': gray,
        'clahe_gray': clahe_img,
        'clahe_rgb': clahe_rgb
    }

    return clahe_rgb_resized, intermediate_steps

@st.cache_resource
def load_model():
    """Ucitaj trenirani model - lokalno ili sa Google Drive-a"""
    local_model_path = './modeli/best_brain_tumor_model.keras'

    # Pokušaj učitati lokalno
    if os.path.exists(local_model_path):
        try:
            model = tf.keras.models.load_model(local_model_path)
            return model
        except Exception as e:
            st.warning(f"Failed to load local model: {e}")

    # Ako lokalni model ne postoji, download sa Google Drive-a
    st.info("Model not found locally. Downloading from Google Drive (first time only)...")

    # Google Drive file ID
    file_id = '1IXILEmBQnwHVqlrE9i1eE1xML0lQuZTE'
    url = f'https://drive.google.com/uc?id={file_id}'

    # Kreiraj direktorijum ako ne postoji
    os.makedirs('./modeli', exist_ok=True)

    # Download model
    try:
        gdown.download(url, local_model_path, quiet=False)
        st.success("Model downloaded successfully!")
        model = tf.keras.models.load_model(local_model_path)
        return model
    except Exception as e:
        st.error(f"Failed to download model from Google Drive: {e}")
        st.error("Please check your Google Drive file ID and sharing permissions.")
        st.stop()

def predict(model, img_array):
    """Predikcija sa modelom"""
    predictions = model.predict(img_array)[0]
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    predicted_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_idx]
    confidence = predictions[predicted_idx] * 100
    
    all_probs = {class_names[i]: predictions[i] * 100 for i in range(len(class_names))}
    
    return predicted_class, confidence, all_probs

# ============= MAIN APP =============
def main():
    st.set_page_config(
        page_title="NeuroVision PRO",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_custom_css()
    
    # Header with embedded button
    header_cols = st.columns([4, 1])

    with header_cols[0]:
        st.markdown("""
        <div class="main-header" style="padding: 20px; margin-bottom: 0;">
            <div class="main-title">NEUROVISION PRO</div>
            <div class="main-subtitle">ADVANCED MRI DIAGNOSTIC SUITE V2.4</div>
        </div>
        """, unsafe_allow_html=True)

    with header_cols[1]:
        st.markdown('<div style="padding-top: 20px;"></div>', unsafe_allow_html=True)

        # Custom styling for this specific button
        st.markdown("""
        <style>
        div[data-testid="column"]:nth-child(2) .stButton button {
            background: linear-gradient(90deg, #00d9ff 0%, #00a8cc 100%);
            color: #000000;
            border: none;
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 0.75rem;
            font-weight: 600;
            width: auto;
            min-width: 80px;
        }
        </style>
        """, unsafe_allow_html=True)

        if st.button("New Case"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Sidebar
    with st.sidebar:
        st.markdown("### System Information")
        st.markdown("""
        <div class="info-card">
            <strong>Model Architecture:</strong> CNN-based<br>
            <strong>Accuracy:</strong> 95.2%<br>
            <strong>Classes:</strong> 4<br>
            <strong>Input Size:</strong> 224x224
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Pre-Analysis Pipeline")
        st.markdown("""
        <div class="info-card">
            <small>STEPS: 04</small>
            <ol style="margin: 10px 0; padding-left: 20px; color: #b8b8b8;">
                <li>Gaussian Sharpening<br><small style="color: #00d9ff;">1.5 × RAW - 0.5 × BLUR</small></li>
                <li>Grayscale Normalization<br><small style="color: #00d9ff;">LUMA COEFFICIENT MAPPING</small></li>
                <li>CLAHE Enhancement<br><small style="color: #00d9ff;">CONTRAST LTD ADAPT EQUALIZATION</small></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.markdown("### Patient Scan Import")
        
        uploaded_file = st.file_uploader(
            "Drop MRI DICOM/PNG/JPG here",
            type=['jpg', 'png', 'jpeg', 'dcm'],
            help="Max file size: 10MB",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Učitaj sliku
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Prikaz originalne slike
            st.markdown("""
            <div class="info-card">
                <p style="color: #00d9ff; margin: 0; font-size: 0.8rem;">RAW INPUT</p>
            </div>
            """, unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            
            # Preprocessing
            st.markdown("### Pre-Analysis Pipeline")
            with st.spinner("Processing scan..."):
                processed_img, steps = preprocess_image(img_array)
            
            # Prikaz preprocessing koraka
            st.markdown('<p style="color: #b8b8b8; font-size: 0.9rem; margin-top: 20px;">STEPS: 04</p>', unsafe_allow_html=True)
            
            cols = st.columns(4)
            step_names = ['GAUSSIAN BLUR', 'HIGH-PASS SHARP', 'GRAYSCALE', 'CLAHE (ENHANCED)']
            step_images = ['blurred', 'sharpened', 'grayscale', 'clahe_gray']
            
            for idx, (col, name, img_key) in enumerate(zip(cols, step_names, step_images)):
                with col:
                    st.markdown(f"""
                    <div class="preprocessing-card">
                        <div class="step-number">{idx+1}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(steps[img_key], use_container_width=True, caption=name)
            
            # Analyze button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("RUN FULL AI ANALYSIS", use_container_width=True, type="primary"):
                st.session_state['run_analysis'] = True
                st.session_state['processed_img'] = processed_img
                # Resize original image za model (224x224)
                original_resized = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_AREA)
                st.session_state['original_img'] = original_resized
    
    with col2:
        if 'run_analysis' in st.session_state and st.session_state['run_analysis']:
            # Raw vs Processed comparison
            st.markdown("### Image Analysis")
            
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.markdown('<p style="color: #b8b8b8; text-align: center;">RAW IMAGE ANALYSIS</p>', unsafe_allow_html=True)
                st.image(st.session_state['original_img'], use_container_width=True)
                st.markdown('<p style="color: #666; text-align: center; font-size: 0.8rem;">Ready for analysis</p>', unsafe_allow_html=True)
            
            with comp_col2:
                st.markdown('<p style="color: #00d9ff; text-align: center;">PROCESSED IMAGE ANALYSIS</p>', unsafe_allow_html=True)
                st.image(st.session_state['processed_img'], use_container_width=True)
                st.markdown('<p style="color: #666; text-align: center; font-size: 0.8rem;">Ready for analysis</p>', unsafe_allow_html=True)
            
            # Prediction
            with st.spinner("Running AI diagnostic model..."):
                model = load_model()
                
                # Normalize i dodaj batch dimension
                img_normalized = st.session_state['processed_img'] / 255.0
                img_batch = np.expand_dims(img_normalized, axis=0)
                
                predicted_class, confidence, all_probs = predict(model, img_batch)
            
            # Results - Improved display
            st.markdown("### Diagnostic Results")

            # Top predictions in a cleaner format
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

            # Main predicted class - highlighted
            st.markdown(f"""
            <div class="result-card" style="border-color: #00ff88; margin-bottom: 20px;">
                <div style="text-align: center;">
                    <p style="color: #666; font-size: 0.9rem; margin: 0; text-transform: uppercase; letter-spacing: 2px;">Predicted Class</p>
                    <p style="color: #00d9ff; font-size: 2.5rem; font-weight: bold; margin: 15px 0;">{sorted_probs[0][0]}</p>
                    <p style="color: #666; font-size: 0.9rem; margin: 0; text-transform: uppercase; letter-spacing: 2px;">Reliability</p>
                    <p style="color: #00ff88; font-size: 2rem; font-weight: bold; margin: 10px 0;">{sorted_probs[0][1]:.1f}%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # All predictions with progress bars
            st.markdown("### Class Probability Distribution")
            for class_name, prob in sorted_probs:
                color = "#00ff88" if class_name == sorted_probs[0][0] else "#00d9ff"
                st.markdown(f"""
                <div style="margin: 15px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="color: {color}; font-weight: bold;">{class_name}</span>
                        <span style="color: {color}; font-weight: bold;">{prob:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(prob / 100))

            # Ground Truth Validation (Optional)
            st.markdown("### Validation")
            st.markdown('<p style="color: #666; font-size: 0.85rem;">Optional: Verify the prediction accuracy by entering the actual diagnosis</p>', unsafe_allow_html=True)

            val_col1, val_col2 = st.columns([3, 1])
            with val_col1:
                ground_truth = st.selectbox(
                    "Actual Diagnosis (Ground Truth)",
                    options=['Select...', 'Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
                    key='ground_truth_select'
                )

            with val_col2:
                st.markdown('<div style="padding-top: 28px;"></div>', unsafe_allow_html=True)
                validate_btn = st.button("Validate", use_container_width=True)

            if validate_btn and ground_truth != 'Select...':
                is_correct = (ground_truth == predicted_class)
                if is_correct:
                    st.markdown(f"""
                    <div class="result-card" style="border-color: #00ff88; background: rgba(0, 255, 136, 0.1);">
                        <p style="color: #00ff88; font-size: 1.2rem; font-weight: bold; text-align: center; margin: 0;">
                             PREDICTION CORRECT
                        </p>
                        <p style="color: #b8b8b8; text-align: center; margin-top: 10px; font-size: 0.9rem;">
                            Model correctly identified: {ground_truth}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card" style="border-color: #ff4444; background: rgba(255, 68, 68, 0.1);">
                        <p style="color: #ff4444; font-size: 1.2rem; font-weight: bold; text-align: center; margin: 0;">
                             PREDICTION INCORRECT
                        </p>
                        <p style="color: #b8b8b8; text-align: center; margin-top: 10px; font-size: 0.9rem;">
                            Predicted: {predicted_class} | Actual: {ground_truth}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        else:
            # Awaiting data state
            st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 400px;">
                <h2 style="color: #b8b8b8; margin-top: 20px;">Awaiting Diagnostic Data</h2>
                <p style="color: #666; text-align: center; max-width: 400px;">
                    Please upload a MRI scan on the left panel to begin the automated preprocessing and classification sequence.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.75rem; margin-top: 50px; padding: 20px; border-top: 1px solid #2a2a2a;">
        <p style="margin: 0 0 8px 0;">AI ASSISTED DIAGNOSTIC TOOL • NOT FOR SOLE CLINICAL USE • COMPLIANT MODULE V2</p>
        <p style="margin: 0; color: #00d9ff; font-size: 0.7rem;">Developed by <strong>Pixel Team</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()