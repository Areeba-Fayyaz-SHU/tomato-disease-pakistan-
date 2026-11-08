import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page setup
st.set_page_config(
    page_title="PakTomato AI | Crop Health Assistant",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Polished UI Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .urdu-font {
        font-family: 'Noto Nastaliq Urdu', serif;
        direction: rtl;
        text-align: right;
        line-height: 2.2;
    }

    .hero-banner {
        background: linear-gradient(135deg, #064e3b 0%, #047857 60%, #10b981 100%);
        border-radius: 18px;
        padding: 2rem 2.5rem;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 12px 30px -10px rgba(4, 120, 87, 0.45);
    }
    
    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .status-pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .pill-danger { background: #fee2e2; color: #991b1b; }
    .pill-warning { background: #fef3c7; color: #92400e; }
    .pill-success { background: #d1fae5; color: #065f46; }

    .protocol-card-ur {
        background: #f0fdf4;
        border-right: 4px solid #059669;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    .protocol-card-en {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 1rem;
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
        "title_ur": "ٹماٹر کا اگیتا جھلساؤ",
        "pathogen": "Alternaria solani",
        "risk": "High Risk",
        "pill": "pill-danger",
        "en": "Prune infected lower foliage immediately to stop spore splash. Apply Chlorothalonil or Mancozeb fungicide spray at early emergence. Strictly avoid overhead spray irrigation.",
        "ur": "نچلے بیمار پتوں کو کاٹ کر تلف کریں۔ مینکوزیب یا کلوروتھیلونل کا سپرے کریں۔ پودوں پر اوپر سے پانی دینے سے گریز کریں۔"
    },
    "Tomato_leaf_late_blight": {
        "title_ur": "ٹماٹر کا پچھیتا جھلساؤ",
        "pathogen": "Phytophthora infestans",
        "risk": "Severe Outbreak",
        "pill": "pill-danger",
        "en": "Highly destructive water mold. Uproot and safely dispose of heavily infected vines. Apply systemic Metalaxyl-M or Dimethomorph fungicides promptly.",
        "ur": "یہ انتہائی تباہ کن بیماری ہے۔ متاثرہ پودوں کو تلف کریں اور فوری طور پر میٹالیکسل زہر کا سپرے کریں۔"
    },
    "Tomato_Septoria_leaf_spot": {
        "title_ur": "سیپٹوریا پتوں کے دھبے",
        "pathogen": "Septoria lycopersici",
        "risk": "Moderate Risk",
        "pill": "pill-warning",
        "en": "Characterized by small circular lesions with dark borders. Increase inter-crop airflow and apply copper-hydroxide formulations early.",
        "ur": "ہوا کی نکاسی بہتر بنانے کے لیے پودوں کا درمیانی فاصلہ رکھیں۔ علامات نظر آتے ہی تانبے والی فنگس کش دوا سپرے کریں۔"
    },
    "Tomato_mold_leaf": {
        "title_ur": "پتوں کی زرد پھپھوندی",
        "pathogen": "Passalora fulva",
        "risk": "Moisture Warning",
        "pill": "pill-warning",
        "en": "Flourishes in excessive canopy humidity. Ensure tunnel cross-ventilation, lower evening irrigation, and utilize sulfur-based protective fungicidal dust.",
        "ur": "فصل اور ٹنلز میں ہوا کی نکاسی یقینی بنائیں۔ شام کے وقت زیادہ پانی دینے سے گریز کریں اور حفاظتی سپرے کریں۔"
    },
    "Tomato_leaf_yellow_curl_virus": {
        "title_ur": "ٹماٹر کا مڑوڑیا وائرس",
        "pathogen": "Begomovirus (Whitefly Vector)",
        "risk": "Viral Threat",
        "pill": "pill-danger",
        "en": "Direct cures do not exist for viral genome infections. Eradicate whitefly insect vectors using Imidacloprid/Acetamiprid applications and yellow sticky card traps.",
        "ur": "اس وائرس کا براہ راست کوئی علاج نہیں۔ سفید مکھی کے خاتمے کے لیے پیلے چپکنے والے کارڈ لگائیں اور امیڈاکلوپرڈ سپرے کریں۔"
    },
    "Tomato_Healthy_leaf": {
        "title_ur": "صحت مند پودا",
        "pathogen": "None (Healthy Foliage)",
        "risk": "Normal State",
        "pill": "pill-success",
        "en": "Tissue exhibits normal vigor. Maintain standard NPK fertilizer ratios and weed control protocols.",
        "ur": "پودا بالکل صحت مند ہے۔ معمول کے مطابق متوازن کھاد اور پانی کا شیڈول جاری رکھیں۔"
    }
}

# Hero Banner
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; align-items: center; gap: 1rem;">
        <span style="font-size: 2.8rem;">🍅</span>
        <div>
            <h1 style="margin: 0; font-size: 1.85rem; font-weight: 700; color: #fff;">
                PakTomato AI &bull; Smart Foliage Diagnostic System
            </h1>
            <p style="margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.95rem;">
                Neural vision engine fine-tuned for local tomato crop pathology across Pakistani agricultural zones.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Layout Setup
col_specimen, col_diagnostics = st.columns([1, 1.2], gap="large")

with col_specimen:
    st.markdown("#### 📷 Leaf Specimen Input")
    uploaded_file = st.file_uploader(
        "Upload high-resolution foliage photograph:",
        type=["jpg", "jpeg", "png"],
        help="Upload an isolated leaf photograph taken under clear daylight for best evaluation accuracy."
    )
    
    if uploaded_file is not None:
        raw_img = Image.open(uploaded_file).convert("RGB")
        st.image(raw_img, caption="Loaded Leaf Specimen", use_container_width=True)
    else:
        st.info("Tip: Capture an individual leaflet centered against field foliage or soil.")

with col_diagnostics:
    st.markdown("#### 🔬 Neural Diagnostic Report")
    
    if uploaded_file is None:
        st.markdown("""
        <div style="border: 2px dashed #cbd5e1; border-radius: 12px; padding: 2.5rem; text-align: center; color: #64748b;">
            <p style="font-size: 1.1rem; margin: 0;">Awaiting leaf upload to initiate neural feature extraction.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("Analyzing foliar lesion patterns..."):
            proc_img = raw_img.resize((224, 224))
            arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(proc_img, dtype=np.float32))
            batch = np.expand_dims(arr, axis=0)
            
            preds = model.predict(batch, verbose=0)[0]
            top_idx = int(np.argmax(preds))
            confidence = float(preds[top_idx])
            pred_class = class_names[top_idx]
            info = ADVISORY_DB.get(pred_class, {})

        readable_name = pred_class.replace('Tomato_', '').replace('_', ' ')
        
        # Primary Diagnostic Card
        st.markdown(f"""
        <div class="card">
            <span class="status-pill {info.get('pill', 'pill-warning')}">{info.get('risk', 'Identified')}</span>
            <h2 style="margin: 0.2rem 0; color: #0f172a; font-size: 1.5rem; text-transform: capitalize;">{readable_name}</h2>
            <p style="margin: 0; color: #64748b; font-size: 0.9rem;"><strong>Pathogen:</strong> <i>{info.get('pathogen', 'N/A')}</i></p>
            <div style="margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.3rem;">
                    <span>Diagnostic Confidence</span>
                    <span style="color: #047857;">{confidence * 100:.2f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(confidence))

        # Advisory Protocols
        st.markdown(f"""
        <div class="protocol-card-ur urdu-font">
            <strong>زرعی ہدایت ({info.get('title_ur', '')}):</strong><br>
            {info.get('ur', '')}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="protocol-card-en">
            <strong style="color: #1e3a8a;">Agronomic Recommendation (English):</strong><br>
            {info.get('en', '')}
        </div>
        """, unsafe_allow_html=True)

        # Full Distribution
        with st.expander("📊 Full Disease Probability Distribution"):
            for i, c_name in enumerate(class_names):
                score = float(preds[i])
                clean_name = c_name.replace("Tomato_", "").replace("_", " ")
                st.write(f"**{clean_name.capitalize()}** — {score*100:.2f}%")
                st.progress(score)
