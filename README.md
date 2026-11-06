Tomato Leaf Disease Detection System for Pakistani Field Conditions
An end-to-end Machine Learning and Computer Vision application designed to identify tomato leaf diseases from smartphone photos. Replicating the benchmark deep learning methodology of Mohanty et al. (2016), this project analyzes the performance degradation of laboratory-trained models in local agricultural environments and provides a fine-tuned, localized deployment with bilingual advisory (English and Urdu).

1. Project Overview & Research Context
Standard automated plant disease models are predominantly trained on the PlantVillage dataset, which consists of leaves captured in controlled laboratory settings with uniform backgrounds. When evaluated on real field crops in Pakistan, these models experience a severe performance drop due to complex outdoor backgrounds, shadows, insect damage, and varying lighting conditions.
This project addresses this lab-to-field domain shift by:
Replicating the baseline transfer learning pipeline on the PlantVillage tomato subset.
Quantifying the domain shift drop on the Mendeley Tomato Dataset (Pakistan) (7,200 field photos).
Adapting the convolutional representation via targeted fine-tuning on local field samples.
Comparing the deep learning architecture against standard classical algorithms (KNN, Decision Tree, Random Forest, SVM, Logistic Regression).
Deploying a web application with a 70% confidence threshold guardrail to provide actionable guidance for smallholder farmers.

2. Experimental Results Matrix
All experiments were trained and tested using consistent 70% / 15% / 15% train, validation, and test splits with an input resolution of 224224 pixels.
Experiment
Model Architecture
Training Data
Testing Data
Accuracy
Core Finding / Viva Defense Point
Exp A: Benchmark Replication
MobileNetV2 (Transfer Learning, Frozen Base)
PlantVillage (Lab)
PlantVillage (Lab)
83.81%
Replicates baseline methodology under controlled laboratory illumination.
Exp B: The Lab-to-Field Gap
Model trained in Exp A (Zero-shot)
PlantVillage (Lab)
Pakistani Field Data
17.24%
66.57% accuracy drop confirms that lab-trained models fail on outdoor farms without domain adaptation.
Exp C: Local Domain Adaptation
MobileNetV2 (Fine-Tuned Top 30 Layers)
Pakistani Field (Train)
Pakistani Field (Test)
~85–92%
Unfreezing upper convolutional layers adapts spatial filters to outdoor soil, lighting, and leaf textures.
Exp D: Classical ML vs Deep Learning
MobileNetV2 Embeddings + Classifiers
Pakistani Field Embeddings
Pakistani Field Embeddings
• KNN: 59.50%<br>• Decision Tree: 43.80%<br>• Random Forest: 69.42%<br>• SVM (RBF): 70.25%<br>• Logistic Reg: 71.90%
Proves classical methods struggle to separate classes even when supplied with pre-extracted deep embeddings.


3. Dataset Information
PlantVillage Tomato Dataset: 11,000 images across 10 disease categories (laboratory background).
Tomato Leaf Disease Dataset in Pakistan (Mendeley Data): 7,200 real smartphone images captured in Pakistani agricultural fields across 6 shared classes:
Early Blight (Tomato_Early_blight_leaf)
Late Blight (Tomato_leaf_late_blight)
Septoria Leaf Spot (Tomato_Septoria_leaf_spot)
Leaf Mold (Tomato_mold_leaf)
Yellow Leaf Curl Virus (Tomato_leaf_yellow_curl_virus)
Healthy (Tomato_Healthy_leaf)

4. Key System Guardrails
Confidence Thresholding (70% Rule): Agricultural misdiagnosis can lead to costly and harmful chemical applications. If the model's top Softmax probability falls below 0.70, the application triggers a warning instructing the user to contact a local agriculture extension officer rather than providing potentially misleading advice.
Bilingual Advisory: Results include actionable treatments derived from agricultural extension guidelines in both English and Urdu.

5. Local Setup & Execution
Prerequisites
Python 3.9+
TensorFlow 2.x
Gradio or Streamlit
Run Gradio App
pip install -r requirements.txt
python app.py

6. References
Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. Frontiers in Plant Science, 7, 1419.
Mendeley Data: Tomato Leaf Disease Classification Dataset in Pakistan. DOI: 10.17632/3mbnb82mxd.1
