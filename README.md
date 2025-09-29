# Emotion Recognition with CNNs (Facial Expression, Valence & Arousal)

This project implements **multi-task learning** for affective computing:

* **Facial Expression Classification** (8 categories)
* **Valence Regression** (continuous: -1 → +1)
* **Arousal Regression** (continuous: -1 → +1)

We experiment with **two backbone CNN architectures**: **EfficientNetB0** and **ResNet50**, using **transfer learning** and **multi-output heads**.

---

## 🚀 Features

* Multi-task CNN (classification + regression)
* Pretrained backbones (EfficientNetB0, ResNet50)
* Advanced data augmentation (rotation, shift, brightness, flip)
* Label smoothing & dropout (reduce overfitting)
* Early stopping + learning rate scheduling
* Evaluation with **classification** and **regression metrics**

---

## 📂 Dataset

* Cropped face images (`224x224`, RGB)
* Annotations:

  * **Expression**: 0 = Neutral, 1 = Happy, 2 = Sad, 3 = Surprise, 4 = Fear, 5 = Disgust, 6 = Anger, 7 = Contempt
  * **Valence**: continuous [-1, +1]
  * **Arousal**: continuous [-1, +1]

Annotations are provided in `.npy` files.

---

## ⚙️ Training Setup

* **Phase 1:** Train heads only (frozen backbone)
* **Phase 2:** Fine-tune backbone + heads
* **Loss Weights:** Expression = 2.0, Valence = 1.0, Arousal = 1.0
* **Optimizers:** Adam (`1e-4` → `2e-5`)

---

## 📊 Results

| Model          | Expression ACC | Valence RMSE | Valence Corr | Arousal RMSE | Arousal Corr |
| -------------- | -------------- | ------------ | ------------ | ------------ | ------------ |
| EfficientNetB0 | 0.16           | 0.52         | 0.13         | 0.42         | 0.03         |
| ResNet50       | 0.28           | 0.45         | 0.36         | 0.39         | 0.18         |

➡️ **ResNet50 outperforms EfficientNetB0** across all tasks, at the cost of longer training.

---

## 🛠 Requirements

* Python 3.8+
* TensorFlow / Keras
* NumPy, Pandas, Matplotlib, Scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📈 Evaluation Metrics

* **Classification:** Accuracy, F1, Cohen’s Kappa, AUC, PR-AUC
* **Regression:** RMSE, Pearson Correlation, SAGR, CCC

---

## 📌 Deliverables

* Jupyter Notebook (`.ipynb`) with modular pipeline
* Training logs + graphs
* Comparison of CNN baselines
* Report (PDF) with discussion and results

---
