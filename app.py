import gradio as gr
import numpy as np
import tensorflow as tf

# Load the localized Pakistani model and classes
model = tf.keras.models.load_model('pk_tomato_model.keras')
with open('pk_classes.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Extension service treatment guide (English and Urdu)
TREATMENT_ADVISORY = {
    "Tomato_Early_blight_leaf": {
        "name_ur": "ٹماٹر کا اگیتا جھلساؤ (Early Blight)",
        "en": "Prune infected lower foliage immediately. Apply Chlorothalonil or Mancozeb fungicide. Avoid overhead watering.",
        "ur": "متاثرہ نچلے پتے فوری طور پر کاٹ کر تلف کریں۔ مینکوزیب یا کلوروتھیلونل کا سپرے کریں۔ پودوں کے اوپر سے پانی دینے سے گریز کریں۔"
    },
    "Tomato_leaf_late_blight": {
        "name_ur": "ٹماٹر کا پچھیتا جھلساؤ (Late Blight)",
        "en": "Severe threat. Destroy infected plants immediately. Spray systemic fungicide containing Metalaxyl or Dimethomorph.",
        "ur": "یہ شدید نقصان دہ بیماری ہے۔ متاثرہ پودے تلف کریں اور فوری طور پر میٹالیکسل یا ڈائیمیتھومورف فنگس کش دوا کا سپرے کریں۔"
    },
    "Tomato_Septoria_leaf_spot": {
        "name_ur": "سیپٹوریا پتوں کے دھبے (Septoria Leaf Spot)",
        "en": "Improve field spacing for airflow. Apply copper-based fungicides at first symptom appearance.",
        "ur": "ہوا کی نکاسی کے لیے پودوں کا درمیانی فاصلہ بہتر بنائیں۔ علامات ظاہر ہوتے ہی تانبے پر مشتمل فنگس کش دوا سپرے کریں۔"
    },
    "Tomato_mold_leaf": {
        "name_ur": "پتوں کی پھپھوندی (Leaf Mold)",
        "en": "Reduce humidity in crop canopy. Ensure adequate cross-ventilation and apply registered preventive fungicides.",
        "ur": "فصل میں نمی کا تناسب کم کریں۔ ہوا کی مناسب نکاسی کو یقینی بنائیں اور حفاظتی سپرے کریں۔"
    },
    "Tomato_leaf_yellow_curl_virus": {
        "name_ur": "ٹماٹر کا وائرس / مڑوڑیا (Yellow Leaf Curl Virus)",
        "en": "Transmitted by whiteflies. Virus cannot be cured directly; control whitefly vector using Imidacloprid or yellow sticky traps.",
        "ur": "یہ وائرس سفید مکھی سے پھیلتا ہے۔ سفید مکھی کے خاتمے کے لیے امیڈاکلوپرڈ سپرے کریں یا پیلے چپکنے والے کارڈز لگائیں۔"
    },
    "Tomato_Healthy_leaf": {
        "name_ur": "صحت مند پتہ (Healthy Leaf)",
        "en": "Leaf shows normal physiological growth. Continue balanced NPK fertilization and clean irrigation.",
        "ur": "پودا بالکل صحت مند ہے۔ مناسب کھاد اور متوازن آبپاشی جاری رکھیں۔"
    }
}

def diagnose(image):
    if image is None:
        return "No image provided", "", ""
    
    img = tf.image.resize(image, (224, 224))
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    preds = model.predict(tf.expand_dims(img_array, 0), verbose=0)[0]
    
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    predicted_class = class_names[top_idx]
    
    # 70% threshold guardrail for decision support
    if confidence < 0.70:
        return (
            "Not Sure / غیر یقینی (<70% Confidence)",
            "The model confidence is low. Please consult an Agriculture Extension Officer instead of taking action.",
            "ماڈل اس نتیجے پر مکمل مطمئن نہیں ہے۔ براہ کرم سپرے سے پہلے قریبی زرعی دفتر کے ماہر سے مشورہ لیں۔"
        )
    
    info = TREATMENT_ADVISORY.get(predicted_class, {
        "name_ur": predicted_class,
        "en": "Follow standard agricultural pest management protocol.",
        "ur": "معیاری زرعی حفاظتی تدابیر پر عمل کریں۔"
    })
    
    diagnosis_str = f"{predicted_class} ({info['name_ur']}) — {confidence * 100:.1f}% Confidence"
    return diagnosis_str, info['en'], info['ur']

# Gradio Interface
demo = gr.Interface(
    fn=diagnose,
    inputs=gr.Image(type="numpy", label="Upload Leaf Photo / تصویر منتخب کریں"),
    outputs=[
        gr.Textbox(label="Diagnosis & Confidence / تشخیص"),
        gr.Textbox(label="Recommended Action (English)"),
        gr.Textbox(label="زرعی ماہرین کی تجویز (Urdu)")
    ],
    title="Tomato Leaf Disease Diagnostic System (Pakistan)",
    description="Fine-tuned deep learning system trained on field-collected tomato foliage from Pakistani crops.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(share=True)
