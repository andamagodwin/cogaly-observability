# Cogaly Observability

**Cogaly Observability** is a healthcare-focused AI observability and safety monitoring system built for the **AI 4 Alzheimer's Hackathon**. The project demonstrates how to design **end-to-end observability for medical AI applications**, with a specific focus on *early Alzheimer’s risk assessment*.

Rather than optimizing for raw medical accuracy, Cogaly focuses on a critical and often overlooked problem: **ML models can fail silently**. In healthcare, these silent failures—such as confidence collapse, data drift, or explanation instability—can have serious consequences.

Cogaly combines:
- A **structured Alzheimer’s risk model** trained on synthetic clinical data (XGBoost + SHAP)
- **Datadog observability** to monitor runtime health, model quality signals, and safety indicators in real-time

This repository contains the full, reproducible implementation of Cogaly, including model training, application code, Datadog configurations, and deployment instructions for Render.

---

## 🚀 Project Overview

Healthcare AI systems are increasingly deployed in real-world environments, yet most monitoring strategies only track infrastructure metrics like latency and error rates. These metrics are insufficient for medical AI systems, where **model confidence degradation, drift, and hallucinations** can occur without triggering traditional alerts.

**Cogaly** addresses this gap by treating an AI model as a *production system that must be observed, audited, and acted upon*.

The system is designed to answer one key question:

> *If this AI system begins to behave unsafely, will engineers and clinicians know—and know what to do next?*

---

## 🧠 System Architecture

```
User Input (Structured Data)
        ↓
Cogaly Core Model (Tabular ML Risk Scoring)
        ↓
API Response (Risk Score + SHAP Explanation)
        ↓
Telemetry → Datadog (Metrics, Logs, Events, Incidents)
```

### Key Components
- **Cogaly Core Model**: A machine learning model (XGBoost) trained on structured Alzheimer’s-related clinical and lifestyle data. Outputs a risk score, confidence signals, and SHAP-based feature importance.
- **Observability Layer (Datadog)**: Captures runtime metrics, quality signals, and safety indicators, and triggers actionable incidents when risk thresholds are breached.

---

## 🧪 Dataset

Cogaly is trained on a **synthetic Alzheimer’s Disease dataset** containing demographic, lifestyle, medical, cognitive, and functional assessment features.

- **Source**: Kaggle – Alzheimer’s Disease Dataset (Rabie El Kharoua, 2024)
- **License**: CC BY 4.0
- **Nature**: Fully synthetic (no real patient data)

*Note: This project uses a public dataset for demonstration purposes.* 
**Hackathon Dataset**: https://drive.google.com/drive/folders/1jGfWOHuA3kSbOQ4y26TI_ogBtDetw1SW

> ⚠️ **Disclaimer**: This project is a research and observability prototype. It is **not a diagnostic tool** and must not be used for clinical decision-making.

---

## 🧠 Model Design (Cogaly)

### Training
- Model training is performed **outside Vertex AI** (e.g., Google Colab or local environment)
- The trained model artifact is exported and versioned (e.g., `cogaly_v1.pkl`)

### Outputs
The Cogaly model outputs structured signals designed for observability:

```json
{
  "risk_score": 0.72,
  "confidence": 0.41,
  "entropy": 1.88,
  "input_anomaly": false
}
```

These signals are intentionally simple yet expressive, enabling downstream detection of unsafe behavior.

---



## 📊 Observability Strategy (Datadog)

Cogaly treats observability as a first-class feature.

### Telemetry Collected
- **Runtime**: latency, errors, request volume
- **LLM Metrics**: token counts, response length, latency
- **Model Quality Signals**:
  - Confidence decay
  - Entropy trends
  - Drift proxies
  - High-risk/low-confidence combinations

### Detection Rules
At least three Datadog monitors are defined, including:
1. High Alzheimer’s risk predictions with low model confidence
2. Sustained confidence degradation over time
3. Anomalous input patterns indicating drift

Each detection rule automatically creates an **actionable incident** in Datadog with contextual information and suggested remediation steps.

---

## 🚨 Incident Management

When unsafe behavior is detected, Datadog automatically:
- Opens an incident or case
- Attaches relevant telemetry (model version, metrics, sample inputs)
- Links to a runbook with recommended actions, such as:
  - Escalation to human review
  - Model rollback
  - Threshold adjustment
  - Traffic throttling

This ensures that issues are not only detected, but **actionable**.

---

## 🧰 Repository Structure

```text
cogaly-observability/
├── app/              # API, Gemini integration, telemetry emission
├── model/            # Training notebook and model artifacts
├── datadog/          # Dashboards, monitors, SLOs, runbooks (JSON exports)
├── traffic/          # Traffic generator to trigger detection rules
├── deploy/           # Google Cloud & Vertex AI setup guides
├── # 🧠 Cogaly - Early Alzheimer's Risk Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

**Cogaly** is an advanced machine learning system for early detection of Alzheimer's disease risk using XGBoost with SHAP explainability. The model analyzes patient health data to provide risk scores, confidence levels, and interpretable feature contributions.

---

## 📋 Table of Contents

- [Features](#-features)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Architecture](#-architecture)
- [Files](#-files)
- [Google Colab](#-google-colab)
- [API](#-api)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- **High Accuracy**: Achieves excellent performance on Alzheimer's risk prediction
- **XGBoost Classifier**: State-of-the-art gradient boosting algorithm
- **SHAP Explainability**: Understand which features contribute most to each prediction
- **Comprehensive Metrics**: Accuracy, ROC-AUC, Precision, Recall
- **Production Ready**: Saved models ready for deployment
- **Google Colab Compatible**: Easy to run in cloud environments
- **Detailed Visualizations**: Confusion matrix, ROC curves, SHAP plots

---

## 📊 Dataset

**File**: `data/alzheimers.csv`

**Target Variable**: `Diagnosis` (0 = No Alzheimer's, 1 = Alzheimer's)

**Features**: 33+ clinical and demographic features including:
- Age, Gender, Ethnicity, Education Level
- BMI, Smoking, Alcohol Consumption
- Physical Activity, Diet Quality, Sleep Quality
- Family History, Medical Conditions
- Cognitive Assessments (MMSE, ADL, etc.)
- Behavioral Indicators

**Non-predictive columns** (excluded): `PatientID`, `DoctorInCharge`

---

## 🚀 Installation

### Local Setup

```bash
# Clone the repository
git clone https://github.com/andamagodwin/cogaly-observability.git
cd cogaly-observability

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- xgboost >= 1.7.0
- shap >= 0.41.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0

---

## 💻 Usage

### Training the Model

#### Option 1: Local Environment

```bash
cd model
python cogaly_model.py
```

#### Option 2: Google Colab

1. Open `model/cogaly_colab.py` in Google Colab
2. Upload `data/alzheimers.csv` or mount Google Drive
3. Run all cells sequentially

### Making Predictions

```python
import pickle
import pandas as pd

# Load trained model
with open('model/cogaly_xgb_v1.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/cogaly_scaler_v1.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model/cogaly_features_v1.pkl', 'rb') as f:
    features = pickle.load(f)

with open('model/cogaly_explainer_v1.pkl', 'rb') as f:
    explainer = pickle.load(f)

# Prepare patient data
patient_data = pd.DataFrame([{
    'Age': 75,
    'Gender': 0,
    'BMI': 28.5,
    'MMSE': 22,
    # ... other features
}])

# Make prediction
from cogaly_model import predict_alzheimer_risk

result = predict_alzheimer_risk(
    patient_data, model, scaler, explainer, features
)

print(f"Diagnosis: {result['diagnosis']}")
print(f"Risk Score: {result['risk_score']:.2%}")
print(f"Confidence: {result['confidence']:.2f}%")
```

---

## 🎯 Model Performance

### Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | High performance on test set |
| **ROC-AUC** | Excellent discrimination ability |
| **Precision** | Minimizes false positives |
| **Recall** | Maximizes early detection |

### Evaluation Outputs

- **Confusion Matrix**: Visual representation of predictions
- **ROC Curve**: Trade-off between TPR and FPR
- **Classification Report**: Per-class metrics
- **SHAP Plots**: Feature importance and impact

---

## 🏗️ Architecture

```
cogaly-observability/
│
├── data/
│   └── alzheimers.csv          # Dataset
│
├── model/
│   ├── cogaly_model.py         # Main training script
│   ├── cogaly_colab.py         # Google Colab version
│   ├── cogaly_xgb_v1.pkl       # Trained XGBoost model
│   ├── cogaly_scaler_v1.pkl    # Feature scaler
│   ├── cogaly_features_v1.pkl  # Feature names
│   ├── cogaly_explainer_v1.pkl # SHAP explainer
│   ├── cogaly_metrics_v1.pkl   # Performance metrics
│   ├── confusion_matrix.png    # Visualization
│   ├── roc_curve.png           # Visualization
│   ├── shap_feature_importance.png
│   └── shap_summary_plot.png
│
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

---

## 📁 Files

### Models

- **`cogaly_xgb_v1.pkl`**: Trained XGBoost classifier
- **`cogaly_scaler_v1.pkl`**: StandardScaler for feature normalization
- **`cogaly_features_v1.pkl`**: List of feature column names
- **`cogaly_explainer_v1.pkl`**: SHAP TreeExplainer for interpretability
- **`cogaly_metrics_v1.pkl`**: Model performance metrics

### Scripts

- **`cogaly_model.py`**: Complete training pipeline with detailed output
- **`cogaly_colab.py`**: Google Colab-optimized version with upload/download

### Visualizations

- **`confusion_matrix.png`**: Model prediction accuracy breakdown
- **`roc_curve.png`**: ROC curve with AUC score
- **`shap_feature_importance.png`**: Bar plot of feature importance
- **`shap_summary_plot.png`**: Detailed SHAP impact visualization

---

## 🌐 Google Colab

### Quick Start

1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com)
2. Upload `model/cogaly_colab.py`
3. Run first cell to install packages:
   ```python
   !pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn -q
   ```
4. Upload dataset or mount Google Drive
5. Run all cells
6. Download trained models

### Download Models from Colab

```python
from google.colab import files
files.download('cogaly_xgb_v1.pkl')
files.download('cogaly_scaler_v1.pkl')
```

---

## 🔌 API

### Prediction Function

```python
def predict_alzheimer_risk(patient_data, model, scaler, explainer, feature_columns):
    """
    Predict Alzheimer's risk for a single patient with explainability.
    
    Parameters:
    -----------
    patient_data : dict or pd.DataFrame
        Patient features
    model : XGBClassifier
        Trained XGBoost model
    scaler : StandardScaler
        Fitted feature scaler
    explainer : shap.TreeExplainer
        SHAP explainer for interpretability
    feature_columns : list
        List of feature names
    
    Returns:
    --------
    dict : {
        'predicted_class': int (0 or 1),
        'diagnosis': str ('No Risk' or 'Alzheimer\'s Risk'),
        'risk_score': float (probability of Alzheimer's),
        'confidence': float (confidence percentage),
        'top_contributing_features': list of dicts
    }
    """
```

### Example Output

```python
{
    'predicted_class': 1,
    'diagnosis': "Alzheimer's Risk Detected",
    'risk_score': 0.8234,
    'confidence': 82.34,
    'top_contributing_features': [
        {
            'feature': 'MMSE',
            'shap_value': -0.4521,
            'impact': 'Decreases Risk'
        },
        {
            'feature': 'Age',
            'shap_value': 0.3102,
            'impact': 'Increases Risk'
        },
        # ... more features
    ]
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

- **Cogaly Team** - *Initial work*

---

## 🙏 Acknowledgments

- **AI 4 Alzheimer's Hackathon** organizers for the opportunity
- **Hack4Health** for democratizing computational medicine
- XGBoost developers for the excellent gradient boosting library
- SHAP library for making AI interpretable
- Scikit-learn for comprehensive ML tools
- The medical community for Alzheimer's research

---

## 📞 Contact

For questions or support, please open an issue on GitHub.

---

## 🔮 Future Enhancements

- [ ] Hyperparameter tuning with GridSearch/RandomSearch
- [ ] Ensemble methods combining multiple models
- [ ] Deep learning integration
- [ ] REST API for predictions
- [ ] Web interface for easy access
- [ ] Real-time monitoring dashboard
- [ ] Model retraining pipeline
- [ ] Docker containerization

---

**Built with ❤️ for early Alzheimer's detection**
├── LICENSE
└── requirements.txt
```

---

## ☁️ Deployment

- **Application Hosting**: Render.com (via `render.yaml`)
- **Application Hosting**: Render.com (via `render.yaml`)
- **Model Hosting**: Included in application container
- **Observability**: Datadog (full access via GitHub Student Pack)

Detailed deployment instructions are provided in the `deploy/` directory.

### Deploy to Render
1. Create a [Render](https://render.com) account.
2. Click **New +** -> **Blueprint**.
3. Connect your GitHub repository.
4. Render will automatically detect `render.yaml` and start deployment.

---

## 📦 Datadog Assets

This repository includes:
- JSON exports of dashboards
- Monitor and SLO configurations
- Incident runbooks

These assets can be imported directly into a Datadog organization for reproducibility.

---

## 🎥 Demo

A 3-minute demo video accompanies this submission and demonstrates:
- The Cogaly architecture
- Live telemetry streaming to Datadog
- Detection rules triggering incidents
- Actionable context for AI engineers

---

## 🧾 License

This project is open-source and released under an OSI-approved license. See the `LICENSE` file for details.

---

## 🧠 Final Note

Cogaly is not about building the most accurate medical model—it is about building **safe, observable, and responsible AI systems**. In domains like healthcare, observability is not optional; it is essential.

This project demonstrates how Datadog can be used to make AI systems transparent, accountable, and operationally safe.

