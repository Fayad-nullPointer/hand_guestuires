# 🖐️ Hand Gesture Recognition

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-enabled-green)
![MLflow](https://img.shields.io/badge/MLflow-tracked-blue?logo=mlflow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A real-time hand gesture recognition system that detects **18 distinct hand gestures** using **MediaPipe** landmark extraction and a **Stacking Ensemble Classifier** (SVM + Random Forest + XGBoost). All experiments are tracked with **MLflow**.

---

## 🎯 Recognized Gestures

| | | | | | |
|---|---|---|---|---|---|
| ☎️ call | 👎 dislike | ✊ fist | 🖐️ four | 👍 like | 🤫 mute |
| 👌 ok | ☝️ one | 🖐️ palm | ✌️ peace | ✌️ peace inv. | 🤘 rock |
| 🛑 stop | 🛑 stop inv. | 🤟 three | 🤟 three2 | ✌️ two up | ✌️ two up inv. |

---

## 📐 How It Works

```
Input Image / Video
        │
        ▼
  MediaPipe HandLandmarker
  (21 3D landmarks per hand)
        │
        ▼
  Custom Coordinate Scaling
  (wrist-relative, normalized by
   middle-finger-tip distance)
        │
        ▼
  Stacking Classifier
  ┌─────────────────────────┐
  │  SVM (RBF, C=100)       │
  │  Random Forest (n=200)  │  ──► Logistic Regression (meta)
  │  XGBoost (n=200)        │
  └─────────────────────────┘
        │
        ▼
  Predicted Gesture Label
  + Temporal Smoothing (mode over 10 frames)
```

### Landmark Scaling

Raw MediaPipe x/y coordinates are normalized to be **position- and scale-invariant**:

$$x'_i = \frac{x_i - x_{wrist}}{x_{mid\_tip} - x_{wrist}}, \quad y'_i = \frac{y_i - y_{wrist}}{y_{mid\_tip} - y_{wrist}}$$

This makes the model robust to hand distance from the camera and hand position in the frame.

---

## 📊 Dataset

- **Total samples:** ~26,000 images (hand landmark coordinates extracted via MediaPipe)
- **Classes:** 18 gesture types
- **Features:** 61 features per sample — 3D coordinates (x, y, z) for 21 landmarks, minus the wrist x/y (used as reference)
- **Split:** 60% train / 20% validation / 20% test (stratified)
- **Class balance:** Approximately balanced (~950–1,650 samples per class)

---

## 🏆 Model Comparison

| Model | Validation Accuracy | Notes |
|---|---|---|
| Logistic Regression | ~85% | Baseline, GD-based |
| Decision Tree | ~88% | With coordinate scaling |
| Random Forest | ~96% | Robust to outliers |
| SVM (RBF, C=100) | ~95% | Without scaling |
| XGBoost | ~96% | With coordinate scaling |
| **Stacking Classifier** | **~97%+** | **Winner 🏆** |

The Stacking Classifier's confusion matrix shows near-perfect diagonal performance across all 18 classes.

---

## 🗂️ Project Structure

```
├── notebook.ipynb              # Full EDA → Modeling → Inference pipeline
├── Mlflow.py                   # MLflow utility functions
├── utils.py                    # MediaPipe drawing helpers
├── hand_landmarks_data.csv     # Extracted landmark dataset
├── hand_landmarker.task        # MediaPipe pre-trained model
├── final_stacking_model.pkl    # Serialized best model
├── mlruns/                     # MLflow experiment logs
└── Test_Viedos/                # Sample test images and videos
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/hand-gesture-recognition.git
cd hand-gesture-recognition
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>Key dependencies</summary>

```
mediapipe>=0.10
scikit-learn>=1.3
xgboost>=2.0
mlflow>=2.0
opencv-python>=4.8
pandas>=2.0
numpy>=1.24
scipy>=1.11
joblib>=1.3
seaborn>=0.12
matplotlib>=3.7
```

</details>

### 3. Download the MediaPipe hand landmarker model

```bash
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

---

## 🚀 Usage

### Run inference on an image

```python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib, pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load model and encoder
model = joblib.load("final_stacking_model.pkl")

# Setup MediaPipe detector
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Detect and predict
image = mp.Image.create_from_file("your_image.jpg")
result = detector.detect(image)
landmarks_df = extract_landmarks_to_df(result)   # see notebook for function
prediction = model.predict(landmarks_df)
```

### Run inference on a video

```python
from notebook import process_video   # or copy the function from the notebook

process_video(
    input_path="input.mp4",
    output_path="output_annotated.mp4",
    pipeline=model,
    label_encoder=le,
    detector=detector,
    num_avg_frames=10   # temporal smoothing window
)
```

---

## 📈 MLflow Experiment Tracking

All runs are tracked locally under `mlruns/`. To view the dashboard:

```bash
mlflow ui
# Open http://localhost:5000
```

Tracked per run:
- Model type & hyperparameters
- Validation accuracy & weighted F1-score
- Confusion matrix (as artifact)
- Serialized model artifact

```python
from Mlflow import track_model, load_best_model, add_artifact

# Log a training run
run_id = track_model(model, "MyModel", X_train, y_train, X_val, y_val, "Hand-Landmark-Recognition")

# Load the best model automatically
best_model = load_best_model(experiment_name="Hand-Landmark-Recognition")
```

---

## 🔀 Branches

| Branch | Purpose |
|---|---|
| `main` | Clean, production-ready code |
| `research` | Full exploration notebook, MLflow runs, grid search experiments |

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push and open a Pull Request against `main`

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) — Hand landmark detection
- [HaGRID Dataset](https://github.com/hukenovs/hagrid) — Hand gesture recognition dataset
- [MLflow](https://mlflow.org/) — Experiment tracking and model registry
