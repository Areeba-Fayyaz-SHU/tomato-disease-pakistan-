import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="PakTomato AI | Crop Health Assistant",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern, Polished Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(45, 106, 79, 0.25);
    }
    
    .metric-card {
        background: #f8faf9;
        border: 1px solid #d8f3dc;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .treatment-box-ur {
        background: #fbfbfb;
        border-right: 5px solid #2d6a4f;
        padding: 1.2rem;
        border-radius: 10px;
        direction: rtl;
        text-align: right;
        font-size: 1.05rem;
        line-height: 1.8;
        color: #1a1a1a;
        margin-top: 1rem;
    }
    
    .treatment-box-en {
        background: #fdfdfd;
        border-left: 5px solid #52b788;
        padding: 1.2rem;
        border-radius: 10px;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #2b2b2b;
        margin-top: 1rem;
    }
    
    .badge-critical {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .badge-safe {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Load Model and Classes with Caching
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('pk_tomato_model.keras')
    with open('pk_classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return model, classes

model, class_names = load_assets()

# Comprehensive Knowledge Base
ADVISORY_DB = {
    "Tomato_Early_blight_leaf": {
        "title_ur": "ٹماٹر کا اگیتا جھلساؤ (Early Blight)",
        "pathogen": "Fungus (Alternaria solani)",
        "status": "High Risk / نقصان دہ",
        "badge_type": "critical",
        "en": "Prune infected lower foliage immediately to stop spore splash. Apply Chlorothalonil or Mancozeb fungicide spray at early emergence. Strictly avoid overhead spray irrigation.",
        "ur": "نچلے بیمار پتوں کو فوری کاٹ کر دور تلف کریں۔ مینکوزیب یا کلوروتھیلونل کا حفاظتی سپرے کریں۔ پودوں کے اوپر سے پانی کے چھڑکاؤ سے مکمل گریز کریں۔"
    },
    "Tomato_leaf_late_blight": {
        "title_ur": "ٹماٹر کا پچھیتا جھلساؤ (Late Blight)",
        "pathogen": "Oomycete (Phytophthora infestans)",
        "status": "Severe Outbreak / تباہ کن",
        "badge_type": "critical",
        "en": "Extremely fast-spreading disease during humid weather. Destroy heavily infected vines immediately. Apply systemic fungicides like Metalaxyl-M or Dimethomorph to protect healthy field sectors.",
        "ur": "یہ بیماری انتہائی تیزی سے پھیلتی ہے۔ متاثرہ پودے نکال کر جلا دیں۔ بیماری کو روکنے کے لیے فوری طور پر میٹالیکسل یا ڈائیمیتھومورف زہر کا سپرے کریں۔"
    },
    "Tomato_Septoria_leaf_spot": {
        "title_ur": "سیپٹوریا پتوں کے دھبے (Septoria Leaf Spot)",
        "pathogen": "Fungus (Septoria lycopersici)",
        "status": "Moderate Risk / درمیانہ خطرہ",
        "badge_type": "critical",
        "en": "Circular gray-centered lesions. Increase plant spacing to promote canopy airflow. Apply registered copper-hydroxide formulations upon first detection.",
        "ur": "ہوا کی نکاسی کے لیے فصل کا درمیانی فاصلہ برقرار رکھیں۔ علامات نظر آتے ہی تانبے (Copper Fungicide) پر مشتمل دوائی کا سپرے کریں۔"
    },
    "Tomato_mold_leaf": {
        "title_ur": "پتوں کی زرد پھپھوندی (Leaf Mold)",
        "pathogen": "Fungus (Passalora fulva)",
        "status": "Moderate Risk / نمی کی بیماری",
        "badge_type": "critical",
        "en": "Common under high humidity conditions. Ensure cross-ventilation in tunnel farming and avoid late evening irrigations. Apply preventive sulfur or bio-fungicide sprays.",
        "ur": "فصل اور سرنگوں (Tunnels) میں ہوا اور دھوپ کی نکاسی بہتر بنائیں۔ شام کے وقت زیادہ پانی دینے سے گریز کریں اور حفاظتی سپرے کریں۔"
    },
    "Tomato_leaf_yellow_curl_virus": {
        "title_ur": "ٹماٹر کا مڑوڑیا وائرس (Yellow Leaf Curl Virus)",
        "pathogen": "Virus (Geminiviridae / Whitefly vector)",
        "status": "Severe Vector Risk / وائرل بیماری",
        "badge_type": "critical",
        "en": "No direct chemical cure exists for the virus. Control the whitefly insect vector immediately using yellow sticky traps and sprays such as Acetamiprid or Imidacloprid.",
        "ur": "اس وائرس کا براہ راست کوئی علاج نہیں ہے۔ اس کو پھیلانے والی سفید مکھی کا خاتمہ کریں۔ کھیت میں پیلے چپکنے والے کارڈ لگائیں اور امیڈاکلوپرڈ کا سپرے کریں۔"
    },
    "Tomato_Healthy_leaf": {
        "title_ur": "صحت مند پودا (Healthy Foliage)",
        "pathogen": "None / صحت مند",
        "status": "Optimal / بہترین حالت",
        "badge_type": "safe",
        "en": "No pathogenic symptoms detected. Continue balanced nitrogen, phosphorus, and potassium (NPK) feeding alongside standard weed management.",
        "ur": "پتے پر کسی قسم کی بیماری کے اثرات نہیں پائے گئے۔ معمول کے مطابق مناسب کھاد اور متوازن آبپاشی کا شیڈول جاری رکھیں۔"
    }
}

# Sidebar - Project Information
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1147/1147805.png", width=70)
    st.markdown("### **PakTomato AI System**")
    st.caption("National Agricultural Decision Support System[cite: 1]")
    st.markdown("---")
    st.markdown("**Field Specification:**")
    st.markdown("- **Engine:** MobileNetV2 (Fine-Tuned)[cite: 1]")
    st.markdown("- **Data Domain:** Pakistani Tomato Foliage[cite: 1]")
    st.markdown("- **Safety Gate:** 70% Confidence Threshold[cite: 1]")
    st.markdown("---")
    st.markdown("ℹ️ *Designed for field testing across agricultural farms in Sindh and Punjab.*")

# Top Header Banner
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700;">🍅 Tomato Crop Health & Diagnostic Engine</h1>
    <p style="margin: 8px 0 0 0; opacity: 0.9; font-size: 1.05rem;">
        Automated plant leaf disease classification and localized field recommendations for Pakistani farmers.[cite: 1]
    </p>
</div>
""", unsafe_allow_html=True)

# Main Two-Column Layout
col_input, col_report = st.columns([1, 1.2], gap="large")

with col_input:
    st.markdown("### 📷 Leaf Specimen Upload")
    input_source = st.radio("Select Image Source:", ["Upload File", "Take Live Camera Shot[cite: 1]"], horizontal=True)
    
    uploaded_file = None
    if input_source == "Upload File":
        uploaded_file = st.file_uploader(
            "Choose a leaf image (PNG, JPG, JPEG):", 
            type=["jpg", "jpeg", "png"],
            help="For best accuracy, upload an isolated, clear leaf image with minimal background clutter.[cite: 1]"
        )
    else:
        uploaded_file = st.camera_input("Capture leaf from field camera")

    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file).convert("RGB")
        st.image(raw_image, caption="Analyzed Leaf Sample", use_container_width=True)

with col_report:
    st.markdown("### 📊 Diagnostic Evaluation")
    
    if uploaded_file is None:
        st.info("👈 Please upload or capture an image of a tomato leaf to generate a comprehensive diagnostic report.")
    else:
        with st.spinner("Processing image via neural network..."):
            # Image preprocessing pipeline
            processed_img = raw_image.resize((224, 224))
            img_arr = tf.keras.applications.mobilenet_v2.preprocess_input(
                np.array(processed_img, dtype=np.float32)
            )
            batch = np.expand_dims(img_arr, axis=0)
            
            # Predict
            raw_preds = model.predict(batch, verbose=0)[0]
            top_idx = int(np.argmax(raw_preds))
            confidence = float(raw_preds[top_idx])
            detected_label = class_names[top_idx]
            info = ADVISORY_DB.get(detected_label, {})

        # Safety Check: Decision Support Threshold
        if confidence < 0.70:
            st.error("⚠️ **Low Confidence Diagnostic Flag (< 70%)**")
            st.markdown("""
            The neural network detected atypical image characteristics, background interference, or indeterminate foliage patterns[cite: 1].
            
            **Safety Action Required:**
            - **English:** Do not apply chemical remedies based on this prediction. Consult your regional Agricultural Extension Officer[cite: 1].
            - **اردو:** ماڈل اس تصویر کی تشخیص پر مکمل مطمئن نہیں ہے۔ براہ کرم زرعی ماہر کی تصدیق کے بغیر کیمیائی اسپرے مت کریں۔
            """)
        else:
            # High-Confidence Diagnostic Report
            badge_class = "badge-safe" if info.get("badge_type") == "safe" else "badge-critical"
            status_tag = info.get("status", "Identified")
            
            st.markdown(f"""
            <div class="metric-card">
                <span class="{badge_class}">{status_tag}</span>
                <h2 style="color: #1b4332; margin-top: 10px; margin-bottom: 5px;">
                    {detected_label.replace('_', ' ')}
                </h2>
                <p style="margin: 0; color: #4b5563; font-size: 0.95rem;">
                    <strong>Pathogen / Type:</strong> {info.get('pathogen', 'N/A')}
                </p>
                <div style="margin-top: 15px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="font-weight: 600;">Model Certainty:</span>
                        <span style="font-weight: 700; color: #2d6a4f;">{confidence * 100:.2f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(float(confidence))

            # Remedial Recommendations
            st.markdown("#### 🌾 Farm Management & Treatment Protocol")
            
            st.markdown(f"""
            <div class="treatment-box-ur">
                <strong>اردو زرعی تجاویز:</strong><br>
                {info.get('ur', 'معیاری زرعی حفاظتی تدابیر پر عمل کریں۔')}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="treatment-box-en">
                <strong>Technical Management (English):</strong><br>
                {info.get('en', 'Follow standard agronomic practices.')}
            </div>
            """, unsafe_allow_html=True)

        # Probability Distribution Accordion
        with st.expander("🔍 View All Disease Probabilities"):
            for i, c_name in enumerate(class_names):
                prob = float(raw_preds[i])
                c_clean = c_name.replace("Tomato_", "").replace("_", " ")
                st.write(f"**{c_clean}**: {prob*100:.2f}%")
                st.progress(prob)
