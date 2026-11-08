import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="PakTomato AI", page_icon="🍅", layout="wide")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        padding: 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8faf9;
        border: 1px solid #d8f3dc;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .treatment-box-ur {
        background: #fff;
        border-right: 5px solid #2d6a4f;
        padding: 1rem;
        border-radius: 8px;
        direction: rtl;
        text-align: right;
        font-size: 1.05rem;
        margin-top: 0.8rem;
    }
    .treatment-box-en {
        background: #fff;
        border-left: 5px solid #52b788;
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.95rem;
        margin-top: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('pk_tomato_model.keras')
    with open('pk_classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return model, classes

model, class_names = load_assets()

ADVISORY_DB = {
    "Tomato_Early_blight_leaf": {
        "title_ur": "ٹماٹر کا اگیتا جھلساؤ (Early Blight)",
        "pathogen": "Alternaria solani",
        "en": "Prune infected lower foliage. Apply Chlorothalonil or Mancozeb fungicide spray.",
        "ur": "نچلے بیمار پتوں کو کاٹ کر تلف کریں۔ مینکوزیب یا کلوروتھیلونل کا سپرے کریں۔"
    },
    "Tomato_leaf_late_blight": {
        "title_ur": "ٹماٹر کا پچھیتا جھلساؤ (Late Blight)",
        "pathogen": "Phytophthora infestans",
        "en": "Severe threat. Destroy heavily infected plants. Spray systemic Metalaxyl fungicide.",
        "ur": "یہ خطرناک بیماری ہے۔ متاثرہ پودے تلف کریں اور فوری میٹالیکسل سپرے کریں۔"
    },
    "Tomato_Septoria_leaf_spot": {
        "title_ur": "سیپٹوریا پتوں کے دھبے (Septoria Leaf Spot)",
        "pathogen": "Septoria lycopersici",
        "en": "Improve plant spacing for airflow. Apply copper-based fungicides at first detection.",
        "ur": "ہوا کی نکاسی کے لیے فاصلہ برقرار رکھیں۔ تانبے پر مشتمل فنگس کش دوا سپرے کریں۔"
    },
    "Tomato_mold_leaf": {
        "title_ur": "پتوں کی زرد پھپھوندی (Leaf Mold)",
        "pathogen": "Passalora fulva",
        "en": "Ensure cross-ventilation in tunnel farming. Apply preventive fungicide sprays.",
        "ur": "ہوا اور دھوپ کی نکاسی بہتر بنائیں۔ شام کو پانی دینے سے گریز کریں۔"
    },
    "Tomato_leaf_yellow_curl_virus": {
        "title_ur": "ٹماٹر کا مڑوڑیا وائرس (Yellow Leaf Curl Virus)",
        "pathogen": "Whitefly Transmitted Virus",
        "en": "Control whitefly vector using Imidacloprid sprays and yellow sticky traps.",
        "ur": "سفید مکھی کے خاتمے کے لیے امیڈاکلوپرڈ سپرے کریں اور پیلے چپکنے والے کارڈ لگائیں۔"
    },
    "Tomato_Healthy_leaf": {
        "title_ur": "صحت مند پودا (Healthy Leaf)",
        "pathogen": "None / صحت مند",
        "en": "No pathogenic symptoms detected. Continue balanced fertilizer and watering schedule.",
        "ur": "پودا بالکل صحت مند ہے۔ مناسب کھاد اور متوازن آبپاشی جاری رکھیں۔"
    }
}

with st.sidebar:
    st.markdown("### **PakTomato AI Settings**")
    threshold = st.slider("Strictness Threshold (%)", min_value=30, max_value=80, value=40, step=5) / 100.0
    st.caption("Lowering this threshold allows the model to display diagnosis on real field photos.")

st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2rem;">🍅 Tomato Crop Health & Diagnostic Engine</h1>
    <p style="margin: 5px 0 0 0; opacity: 0.9;">Trained on real Pakistani field foliage.</p>
</div>
""", unsafe_allow_html=True)

col_input, col_report = st.columns([1, 1.2], gap="large")

with col_input:
    st.markdown("### 📷 Upload Leaf Specimen")
    uploaded_file = st.file_uploader("Upload leaf image:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file).convert("RGB")
        st.image(raw_image, caption="Analyzed Leaf", use_container_width=True)

with col_report:
    st.markdown("### 📊 Diagnostic Evaluation")
    if uploaded_file is None:
        st.info("👈 Please upload an image of a leaf to see the disease diagnosis.")
    else:
        with st.spinner("Analyzing leaf patterns..."):
            processed_img = raw_image.resize((224, 224))
            img_arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(processed_img, dtype=np.float32))
            batch = np.expand_dims(img_arr, axis=0)
            
            raw_preds = model.predict(batch, verbose=0)[0]
            top_idx = int(np.argmax(raw_preds))
            confidence = float(raw_preds[top_idx])
            detected_label = class_names[top_idx]
            info = ADVISORY_DB.get(detected_label, {})

        # Display Top Diagnosis regardless of threshold
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #1b4332; margin: 0 0 5px 0;">{detected_label.replace('_', ' ')}</h3>
            <p style="margin: 0 0 10px 0; color: #555;"><strong>Pathogen:</strong> {info.get('pathogen', 'N/A')}</p>
            <div style="display: flex; justify-content: space-between;">
                <span><strong>Prediction Confidence:</strong></span>
                <span style="font-weight: 700; color: #2d6a4f;">{confidence * 100:.2f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(confidence))

        # Check safety warning
        if confidence < threshold:
            st.warning(f"⚠️ **Moderate Confidence ({confidence*100:.1f}%)**: The model suggests this disease, but recommends confirming with a field expert.")
        else:
            st.success(f"✅ **High Confidence Identification ({confidence*100:.1f}%)**")

        st.markdown(f"""
        <div class="treatment-box-ur">
            <strong>اردو زرعی تجاویز ({info.get('title_ur', '')}):</strong><br>
            {info.get('ur', '')}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="treatment-box-en">
            <strong>Management Protocol (English):</strong><br>
            {info.get('en', '')}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍 View All Disease Probabilities"):
            for i, c_name in enumerate(class_names):
                p = float(raw_preds[i])
                clean_name = c_name.replace("Tomato_", "").replace("_", " ")
                st.write(f"**{clean_name}**: {p*100:.2f}%")
                st.progress(p)
