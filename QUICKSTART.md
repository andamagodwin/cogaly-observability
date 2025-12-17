# Cogaly Model - Quick Start Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

```bash
# Navigate to project directory
cd cogaly-observability

# Install dependencies
pip install -r requirements.txt
```

### Running the Model

```bash
# Train the model
cd model
python cogaly_model.py
```

### Expected Output

The script will:
1. ✅ Load and preprocess the dataset (2,149 patients)
2. ✅ Split data into 80/20 train/test sets
3. ✅ Train XGBoost classifier
4. ✅ Evaluate model performance
5. ✅ Generate SHAP explanations
6. ✅ Save trained model and artifacts

### Model Performance

**Achieved Results:**
- **Accuracy**: 94.42%
- **ROC-AUC**: 0.9480
- **Precision**: 93.84%
- **Recall**: 90.13%

### Output Files

After training, you'll find:

**Models:**
- `cogaly_xgb_v1.pkl` - Trained XGBoost model
- `cogaly_scaler_v1.pkl` - Feature scaler
- `cogaly_feature_columns_v1.pkl` - Feature names
- `cogaly_shap_explainer_v1.pkl` - SHAP explainer
- `cogaly_metrics_v1.pkl` - Performance metrics

**Visualizations:**
- `confusion_matrix.png` - Classification accuracy
- `roc_curve.png` - ROC curve with AUC score
- `shap_feature_importance.png` - Feature importance
- `shap_summary_plot.png` - Detailed SHAP analysis

## 📊 Using the Model

### Load Trained Model

```python
import pickle
import pandas as pd

# Load all components
with open('model/cogaly_xgb_v1.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/cogaly_scaler_v1.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model/cogaly_features_v1.pkl', 'rb') as f:
    features = pickle.load(f)

with open('model/cogaly_explainer_v1.pkl', 'rb') as f:
    explainer = pickle.load(f)
```

### Make Predictions

```python
# Prepare patient data
patient = pd.DataFrame([{
    'Age': 75,
    'Gender': 0,
    'BMI': 28.5,
    'MMSE': 22.0,
    'FunctionalAssessment': 7.5,
    'ADL': 5.2,
    # ... include all 32 features
}])

# Import prediction function
from cogaly_model import predict_alzheimer_risk

# Get prediction with explainability
result = predict_alzheimer_risk(
    patient, model, scaler, explainer, features
)

# View results
print(f"Diagnosis: {result['diagnosis']}")
print(f"Risk Score: {result['risk_score']:.2%}")
print(f"Confidence: {result['confidence']:.1f}%")
print("\nTop Contributing Features:")
for feat in result['top_features']:
    print(f"  - {feat['feature']}: {feat['impact']}")
```

## 🌐 Google Colab

### Setup in Colab

```python
# Install packages
!pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn -q

# Upload dataset
from google.colab import files
uploaded = files.upload()  # Upload alzheimers.csv

# Run the script
!python cogaly_colab.py

# Download trained models
from google.colab import files
files.download('cogaly_xgb_v1.pkl')
files.download('cogaly_scaler_v1.pkl')
```

## 🔍 Top Contributing Features

Based on SHAP analysis:

1. **FunctionalAssessment** (Importance: 2.32)
2. **ADL** (Activities of Daily Living) (Importance: 2.06)
3. **MemoryComplaints** (Importance: 1.57)
4. **MMSE** (Mini-Mental State Exam) (Importance: 1.41)
5. **BehavioralProblems** (Importance: 1.34)

## 📈 Model Architecture

```
Input Features (32)
        ↓
StandardScaler (Normalization)
        ↓
XGBoost Classifier
  ├─ Default hyperparameters
  ├─ Random state: 42
  └─ Eval metric: logloss
        ↓
Predictions + Probabilities
        ↓
SHAP Explainer
        ↓
Risk Score + Top Features
```

## 🛠️ Troubleshooting

### Common Issues

**Issue: Module not found**
```bash
pip install --upgrade pandas numpy scikit-learn xgboost shap matplotlib seaborn
```

**Issue: Memory error**
- Reduce dataset size for testing
- Use a machine with more RAM
- Run in Google Colab (free GPU/RAM)

**Issue: SHAP computation slow**
- Normal for large datasets
- Can skip SHAP for faster training (comment out Step 9)

## 📝 Dataset Requirements

Your dataset must include these features:
- Age, Gender, Ethnicity, EducationLevel
- BMI, Smoking, AlcoholConsumption
- PhysicalActivity, DietQuality, SleepQuality
- Family history and medical conditions
- Cognitive assessments (MMSE, ADL, etc.)
- Behavioral indicators
- **Target**: Diagnosis (0 or 1)

## 🎯 Next Steps

1. **Hyperparameter Tuning**: Improve performance with GridSearch
2. **Feature Engineering**: Create new predictive features
3. **Ensemble Methods**: Combine multiple models
4. **Deployment**: Create REST API or web interface
5. **Monitoring**: Track model performance over time

## 📞 Support

For issues or questions:
1. Check the README.md
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Open an issue on GitHub

## ✅ Verification Checklist

- [ ] Python 3.8+ installed
- [ ] All dependencies installed
- [ ] Dataset (alzheimers.csv) in data/ folder
- [ ] Script runs without errors
- [ ] Model files generated in model/ folder
- [ ] Performance metrics are satisfactory
- [ ] SHAP plots generated successfully

---

**Ready to detect Alzheimer's risk with AI! 🧠💙**
