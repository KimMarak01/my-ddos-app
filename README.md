# 💻 DDoS Detection ML App

This is a **Streamlit web application** for detecting DDoS attacks using machine learning models. Users can upload a CSV dataset, select a model, and get predictions along with performance metrics.

---

## 🚀 Features

- Upload your own CSV dataset for predictions
- Choose between multiple ML models:
  - Random Forest
  - Logistic Regression
  - SVM (RBF)
- Preprocessing of input data (handling NaNs, infinities, and scaling)
- Predictions displayed in a table
- Model metrics (accuracy, classification report, confusion matrix) if true labels are provided

---

## 📦 Files

- `app.py` – Main Streamlit app
- `scaler.joblib` – Pretrained StandardScaler
- `random_forest_model.joblib` – Random Forest model
- `logistic_regression_model.joblib` – Logistic Regression model
- `svm_model.joblib` – Support Vector Machine model
- `label_encoder.joblib` – Label encoder for target classes
- `requirements.txt` – Required Python packages

> Large model files are tracked using **Git LFS**.

---


So your full section would look like:

```markdown
## 🖥️ How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/KimMarak01/my-ddos-app.git
cd my-ddos-app
pip install -r requirements.txt
pip install -r requirements.txt
