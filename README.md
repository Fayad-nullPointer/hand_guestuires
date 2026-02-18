# Hand Recognition Project

## 1. What is the Project About?
This project focuses on **hand gesture recognition** using machine learning and computer vision. The goal is to classify hand signs from images or videos, enabling applications such as sign language interpretation, gesture-based control, and human-computer interaction.

---

## 2. Architecture Overview
- **Data Collection:** Hand landmark data is collected and labeled.
- **Preprocessing:** Data is cleaned, scaled, and encoded.
- **Modeling:** Multiple machine learning models are trained and evaluated.
- **Tracking:** MLflow is used for experiment tracking and model management.
- **Integration:** The best model is integrated with MediaPipe for real-time hand landmark detection and OpenCV for visualization.
- **Deployment:** The system processes video input and outputs annotated video with predicted hand signs.

---

## 3. Models Tried
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **Decision Tree**
- **XGBoost**
- **Stacking Classifier** (ensemble of SVM, Random Forest, XGBoost with Logistic Regression as meta-learner)

---

## 4. MLflow Tracking
- All experiments, hyperparameters, and metrics are tracked using **MLflow**.
- Best models are registered and loaded directly from the MLflow model registry.
- Artifacts such as confusion matrices are logged for each run.

---

## 5. Dataset
- The dataset consists of hand landmark coordinates (x, y, z) for 21 keypoints per hand, extracted from images.
- Each sample is labeled with the corresponding hand sign/class.
- Data is split into training, validation, and test sets, with checks for class imbalance.

---

## 6. Data Preprocessing
- **Scaling:** Landmarks are normalized relative to the wrist and middle finger tip to ensure invariance to hand position and size.
- **Encoding:** Labels are encoded using `LabelEncoder`.
- **Outlier Handling:** Visualization and robust models are used to mitigate the effect of outliers.
- **Feature Engineering:** Only relevant features are kept after scaling.

---

## 7. MediaPipe Integration
- **MediaPipe** is used for real-time hand landmark detection from images and videos.
- Detected landmarks are processed and fed into the trained ML pipeline for gesture classification.
- **OpenCV** is used for drawing landmarks, connections, and predicted labels on video frames.

---

## 8. License
This project is released under the **MIT License**. Please see the `LICENSE` file for details.

---

## 9. Collaboration & Final Statement
This project is open for collaboration! Contributions, suggestions, and improvements are welcome. The combination of robust ML modeling, experiment tracking, and real-time computer vision integration makes this project a strong foundation for further research and practical applications in gesture recognition.

---
