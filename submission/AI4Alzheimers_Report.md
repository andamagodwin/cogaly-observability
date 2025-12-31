# Cogaly: Early Alzheimer's Risk Detection Using XGBoost with SHAP Explainability

**AI 4 Alzheimer's Hackathon Submission**  
**Team:** Cogaly  
**Date:** December 29, 2025

---

## 1. Problem Framing

### 1.1 Background

Alzheimer's disease affects over 55 million people worldwide, with projections suggesting this number could triple by 2050. Early detection is crucial for:
- Enabling timely medical intervention
- Improving patient outcomes and quality of life
- Reducing healthcare costs through proactive care
- Supporting families in care planning

### 1.2 Problem Statement

Current clinical diagnosis of Alzheimer's often occurs late in disease progression, when significant cognitive decline has already occurred. We need **accessible, interpretable AI tools** that can:
1. Identify at-risk individuals using routine clinical and lifestyle data
2. Provide transparent, explainable predictions for clinician trust
3. Highlight the key contributing factors for each patient

### 1.3 Our Solution: Cogaly

**Cogaly** is an XGBoost-based machine learning system that predicts Alzheimer's risk using 32 clinical, demographic, and lifestyle features.

Unlike traditional models, Cogaly prioritizes **Safety and Observability**. It integrates:
1.  **SHAP (SHapley Additive exPlanations)** for per-prediction interpretability.
2.  **Datadog Observability** for real-time monitoring of model confidence, drift, and operational health.
3.  **Cloud Native Deployment** on Render for continuous delivery.

---

## 2. Methods

### 2.1 Dataset

- **Source:** Synthetic Alzheimer's Disease Dataset (Kaggle)
- **Size:** 2,149 patient records with 35 features
- **Target Variable:** Binary diagnosis (0 = No Alzheimer's, 1 = Alzheimer's)
- **Class Distribution:** 64.6% No Risk, 35.4% At Risk

**Feature Categories:**
| Category | Features |
|----------|----------|
| Demographics | Age, Gender, Ethnicity, Education Level |
| Lifestyle | BMI, Smoking, Alcohol, Physical Activity, Diet Quality, Sleep Quality |
| Medical History | Family History, Cardiovascular Disease, Diabetes, Depression, Hypertension |
| Clinical Measurements | Systolic/Diastolic BP, Cholesterol (Total, LDL, HDL, Triglycerides) |
| Cognitive Assessment | MMSE, Functional Assessment, ADL (Activities of Daily Living) |
| Behavioral Indicators | Memory Complaints, Confusion, Disorientation, Personality Changes |

### 2.2 Preprocessing Pipeline

1. **Data Cleaning:** Removed non-predictive columns (`PatientID`, `DoctorInCharge`)
2. **Missing Values:** Checked and handled (none detected in this dataset)
3. **Feature Encoding:** Applied one-hot encoding for categorical variables
4. **Feature Scaling:** StandardScaler normalization (mean=0, std=1)
5. **Train-Test Split:** 80/20 stratified split to preserve class balance

### 2.3 Model Architecture

**Algorithm:** XGBoost (eXtreme Gradient Boosting) Classifier

```
Input Features (32)
       ↓
StandardScaler (Normalization)
       ↓
XGBoost Classifier
  ├─ Ensemble of decision trees
  ├─ Gradient boosting optimization
  └─ Log-loss evaluation metric
       ↓
Risk Probability + Binary Prediction
       ↓
SHAP TreeExplainer
       ↓
Interpretable Feature Contributions
```

**Why XGBoost?**
- Excellent performance on tabular/clinical data
- Handles feature interactions naturally
- Fast training and inference
- Native integration with SHAP for explainability

### 2.4 Explainability with SHAP

For each prediction, we compute SHAP values that show:
- **Which features** contributed to the prediction
- **Magnitude** of each feature's impact
- **Direction** (increases or decreases risk)

This transparency is essential for clinical adoption and regulatory compliance.

### 2.5 Observability & Deployment

**Deployment Architecture:**
- **Platform:** Render (PaaS)
- **Container:** Dockerized Python 3.11 Environment
- **API:** FastAPI (Asynchronous inference)

**Observability (Datadog):**
We implemented a "Safety Layer" using Datadog to detect silent model failures:
- **Confidence Monitoring:** Alerts if average prediction confidence drops below 75%.
- **Latency Tracking:** Ensures predictions are delivered <200ms.
- **Drift Detection:** Monitors distribution of key features (e.g., Age, MMSE) for deviations from training data.

---

## 3. Evaluation

### 3.1 Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 94.42% |
| **ROC-AUC** | 0.9480 |
| **Precision** | 93.84% |
| **Recall** | 90.13% |
| **F1-Score** | 0.92 |

### 3.2 Confusion Matrix

|                     | Predicted: No Risk | Predicted: Risk |
|---------------------|--------------------:|----------------:|
| **Actual: No Risk** | 269 (TN) | 9 (FP) |
| **Actual: Risk**    | 15 (FN) | 137 (TP) |

**Clinical Implications:**
- **90.13% Recall:** Catches 9 out of 10 at-risk patients
- **93.84% Precision:** Only 3.2% false alarm rate
- **Low False Negative Rate:** Prioritizes early detection

### 3.3 Top Contributing Features (SHAP Analysis)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Functional Assessment | 2.32 |
| 2 | ADL (Activities of Daily Living) | 2.06 |
| 3 | Memory Complaints | 1.57 |
| 4 | MMSE (Cognitive Test) | 1.41 |
| 5 | Behavioral Problems | 1.34 |

These findings align with clinical knowledge—cognitive assessments (MMSE, ADL) and behavioral indicators are the strongest predictors.

### 3.4 Visualization

The notebook includes:
- **Confusion Matrix Heatmap**
- **ROC Curve** (AUC = 0.948)
- **SHAP Feature Importance Plot**
- **SHAP Summary Plot** (detailed impact visualization)

---

## 4. Conclusions & Future Work

### 4.1 Key Contributions

1. **High-Performance Model:** 94.42% accuracy with strong recall for early detection
2. **Interpretable Predictions:** SHAP integration provides transparent, clinician-friendly explanations
3. **Safe & Observable:** Integrated Datadog monitoring to detect drift and confidence collapse
4. **Reproducible Pipeline:** Complete, well-documented notebook and automated Render deployment

### 4.2 Limitations

- Trained on synthetic data; requires validation on real clinical datasets
- Single time-point analysis; longitudinal tracking could improve predictions
- Missing advanced biomarkers (genetic markers, brain imaging)

### 4.3 Future Directions

- Hyperparameter tuning with cross-validation
- Ensemble methods combining multiple algorithms
- Integration of imaging and genetic data
- Deployment as a clinical decision support API

---

## 5. Reproducibility

**GitHub Repository:** [cogaly-observability](https://github.com/andamagodwin/cogaly-observability)

**To reproduce:**
```bash
# Clone repository
git clone https://github.com/andamagodwin/cogaly-observability.git
cd cogaly-observability

# Install dependencies
pip install -r requirements.txt

# Run training
python model/cogaly_model.py
```

Or open the provided **Google Colab notebook** and run all cells.

---

## References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
2. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
3. Alzheimer's Association. (2024). Alzheimer's Disease Facts and Figures.

---

**Disclaimer:** This model is intended for research and educational purposes. It is not approved for clinical diagnosis. Always consult qualified healthcare professionals for medical decisions.
