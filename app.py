
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Tomato Disease Detector PK", layout="centered")

@st.cache_resource
def load_disease_model():
    model = tf.keras.models.load_model('pk_tomato_model.keras')
    with open('pk_classes.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return model, classes

model, class_names = load_disease_model()

TREATMENT_ADVISORY = {
    "Tomato_Early_blight_leaf": {
        "ur_title": "ٹماٹر کا اگیتا جھلساؤ",
        "en": "Prune infected lower foliage immediately. Apply Chlorothalonil or Mancozeb fungicide. Avoid overhead watering.",
        "ur": "متاثرہ نچلے پتے فوری طور پر کاٹ کر تلف کریں۔ مینکوزیب یا کلوروتھیلونل کا سپرے کریں۔ پودوں کے اوپر سے پانی دینے سے گریز کریں۔"
    },
    "Tomato_leaf_late_blight": {
        "ur_title": "ٹماٹر کا پچھیتا جھلساؤ",
        "en": "Severe threat. Destroy infected plants immediately. Spray systemic fungicide containing Metalaxyl or Dimethomorph.",
        "ur": "یہ شدید نقصان دہ بیماری ہے۔ متاثرہ پودے تلف کریں اور فوری طور پر میٹالیکسل یا ڈائیمیتھومورف سپرے کریں۔"
    },
    "Tomato_Septoria_leaf_spot": {
        "ur_title": "سیپٹوریا پتوں کے دھبے",
        "en": "Improve field spacing for airflow. Apply copper-based fungicides at first symptom appearance.",
        "ur": "ہوا کی نکاسی کے لیے پودوں کا درمیانی فاصلہ بہتر بنائیں۔ تانبے پر مشتمل فنگس کش دوا سپرے کریں۔"
    },
    "Tomato_mold_leaf": {
        "ur_title": "پتوں کی پھپھوندی",
        "en": "Reduce humidity in crop canopy. Ensure adequate cross-ventilation and apply registered preventive fungicides.",
        "ur": "فصل میں نمی کا تناسب کم کریں۔ ہوا کی مناسب نکاسی کو یقینی بنائیں اور حفاظتی سپرے کریں۔"
    },
    "Tomato_leaf_yellow_curl_virus": {
        "ur_title": "ٹماٹر کا وائرس / مڑوڑیا",
        "en": "Transmitted by whiteflies. Virus cannot be cured directly; control whitefly vector using Imidacloprid or yellow sticky traps.",
        "ur": "یہ وائرس سفید مکھی سے پھیلتا ہے۔ سفید مکھی کے خاتمے کے لیے امیڈاکلوپرڈ سپرے کریں یا پیلے چپکنے والے کارڈز لگائیں۔"
    },
    "Tomato_Healthy_leaf": {
        "ur_title": "صحت مند پتہ",
        "en": "Leaf shows normal physiological growth. Continue balanced NPK fertilization and clean irrigation.",
        "ur": "پودا بالکل صحت مند ہے۔ مناسب کھاد اور متوازن دیکھ بھال جاری رکھیں۔"
    }
}

st.title("🍅 Tomato Leaf Disease Detector (Pakistan)")
st.write("Deep learning diagnostic model fine-tuned on Pakistani field conditions.")

uploaded_file = st.file_uploader("Upload Leaf Photo / پتے کی تصویر منتخب کریں", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Leaf Image", use_column_width=True)
    
    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img_resized, dtype=np.float32))
    img_batch = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_batch, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    conf = float(preds[top_idx])
    predicted_class = class_names[top_idx]
    
    st.subheader("Diagnostic Results / نتائج")
    if conf < 0.70:
        st.warning("⚠️ **Low Confidence / غیر یقینی (<70%)**")
        st.write("Model confidence is too low to guarantee diagnosis. Please consult an Agriculture Extension Officer.")
        st.write("ماڈل اس نتیجے پر مکمل مطمئن نہیں ہے۔ برائے مہربانی قریبی محکمہ زراعت کے ماہر سے مشورہ لیں۔")
    else:
        st.success(f"**Diagnosis:** {predicted_class} ({conf*100:.1f}%)")
        adv = TREATMENT_ADVISORY.get(predicted_class, {"ur_title": "", "en": "Consult agriculture officer.", "ur": "زرعی ماہر سے رابطہ کریں۔"})
        st.info(f"**اردو نام:** {adv['ur_title']}")
        st.markdown(f"**Recommended Action (English):** {adv['en']}")
        st.markdown(f"**زرعی ماہرین کی تجویز (Urdu):** {adv['ur']}")
