"""
🧠 COGALY - ALZHEIMER'S RISK DETECTION MODEL
============================================
Google Colab Ready Version

Instructions to run in Google Colab:
1. Upload 'alzheimers.csv' to Colab or mount Google Drive
2. Run all cells sequentially
3. Model will be trained and saved automatically

Author: Cogaly Team
Date: December 17, 2025
"""

# ============================================================================
# STEP 1: INSTALL REQUIRED PACKAGES (Run this cell first in Colab)
# ============================================================================

# Uncomment the following lines when running in Google Colab
# !pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn -q
# print("✅ All packages installed successfully!")

# ============================================================================
# STEP 2: IMPORT LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve
)
import xgboost as xgb
import shap
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🧠 COGALY - ALZHEIMER'S RISK DETECTION SYSTEM")
print("=" * 80)
print("\n✅ All libraries imported successfully!\n")

# ============================================================================
# STEP 3: UPLOAD DATA (For Google Colab)
# ============================================================================

# Option 1: Upload file directly in Colab
# Uncomment the following lines to upload file:
"""
from google.colab import files
uploaded = files.upload()
data_path = 'alzheimers.csv'
"""

# Option 2: Load from Google Drive
# Uncomment the following lines to use Google Drive:
"""
from google.colab import drive
drive.mount('/content/drive')
data_path = '/content/drive/MyDrive/path/to/alzheimers.csv'
"""

# For local testing, use the relative path
data_path = '../data/alzheimers.csv'  # Change this path as needed

print("📁 Data path configured")

# ============================================================================
# STEP 4: LOAD AND EXPLORE DATA
# ============================================================================

print("\n" + "=" * 80)
print("📊 LOADING DATASET")
print("=" * 80)

df = pd.read_csv(data_path)

print(f"✅ Dataset loaded successfully!")
print(f"   • Total samples: {df.shape[0]:,}")
print(f"   • Total features: {df.shape[1]}")
print(f"\n📋 First few rows:")
display(df.head())

print(f"\n📈 Target Distribution:")
target_dist = df['Diagnosis'].value_counts()
print(target_dist)
print(f"\nClass Balance:")
print(df['Diagnosis'].value_counts(normalize=True) * 100)

# ============================================================================
# STEP 5: DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("🔧 DATA PREPROCESSING")
print("=" * 80)

# Remove non-predictive columns
columns_to_drop = ['PatientID', 'DoctorInCharge']
df_processed = df.drop(columns=columns_to_drop, errors='ignore')
print(f"✅ Removed columns: {columns_to_drop}")

# Separate features and target
X = df_processed.drop('Diagnosis', axis=1)
y = df_processed['Diagnosis']
print(f"✅ Features: {X.shape[1]}, Target: Diagnosis")

# Handle missing values
missing_count = X.isnull().sum().sum()
if missing_count > 0:
    print(f"⚠️  Handling {missing_count} missing values...")
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].mode()[0], inplace=True)
    print(f"✅ Missing values handled")
else:
    print(f"✅ No missing values detected")

# Encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print(f"✅ Encoding {len(categorical_cols)} categorical features...")
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print(f"\n✅ Preprocessing complete! Final features: {X.shape[1]}")

# ============================================================================
# STEP 6: TRAIN-TEST SPLIT
# ============================================================================

print("\n" + "=" * 80)
print("✂️  SPLITTING DATA (80/20)")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✅ Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# ============================================================================
# STEP 7: FEATURE SCALING
# ============================================================================

print("\n" + "=" * 80)
print("📏 FEATURE SCALING")
print("=" * 80)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), 
    columns=X_train.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), 
    columns=X_test.columns
)

print(f"✅ Features scaled (mean=0, std=1)")

# ============================================================================
# STEP 8: TRAIN XGBOOST MODEL
# ============================================================================

print("\n" + "=" * 80)
print("🤖 TRAINING XGBOOST CLASSIFIER")
print("=" * 80)

model = xgb.XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

print("⏳ Training in progress...")
model.fit(X_train_scaled, y_train)
print("✅ Model training completed!")

# ============================================================================
# STEP 9: MAKE PREDICTIONS
# ============================================================================

print("\n" + "=" * 80)
print("🔮 MAKING PREDICTIONS")
print("=" * 80)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

print(f"✅ Predictions generated for {len(y_test):,} samples")

# ============================================================================
# STEP 10: EVALUATE MODEL
# ============================================================================

print("\n" + "=" * 80)
print("📊 MODEL EVALUATION")
print("=" * 80)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\n🎯 PERFORMANCE METRICS:")
print(f"{'='*80}")
print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   ROC-AUC:   {roc_auc:.4f}")
print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"{'='*80}")

print(f"\n📋 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, 
                          target_names=['No Alzheimer\'s', 'Alzheimer\'s']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n🔢 CONFUSION MATRIX:")
print(f"   True Negatives:  {cm[0,0]:,}")
print(f"   False Positives: {cm[0,1]:,}")
print(f"   False Negatives: {cm[1,0]:,}")
print(f"   True Positives:  {cm[1,1]:,}")

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['No Risk', 'Risk'],
           yticklabels=['No Risk', 'Risk'])
plt.title('Confusion Matrix - Cogaly Model', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
        label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
        label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Cogaly Model', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# STEP 11: SHAP EXPLAINABILITY
# ============================================================================

print("\n" + "=" * 80)
print("🔍 SHAP EXPLAINABILITY ANALYSIS")
print("=" * 80)

print("⏳ Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)
print("✅ SHAP values computed!")

# Feature importance
shap_importance = pd.DataFrame({
    'Feature': X_test_scaled.columns,
    'Importance': np.abs(shap_values).mean(axis=0)
}).sort_values('Importance', ascending=False)

print(f"\n🏆 TOP 10 CONTRIBUTING FEATURES:")
print("-" * 80)
for idx, (_, row) in enumerate(shap_importance.head(10).iterrows(), 1):
    print(f"   {idx:2d}. {row['Feature']:30s} | {row['Importance']:.6f}")

# SHAP Bar Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", show=False)
plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# SHAP Summary Plot
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test_scaled, show=False)
plt.title('SHAP Impact on Predictions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# STEP 12: PREDICTION FUNCTION
# ============================================================================

def predict_alzheimer_risk(patient_data, model, scaler, explainer, feature_cols):
    """
    Predict Alzheimer's risk with explainability
    
    Returns: Dictionary with prediction, risk score, confidence, and top features
    """
    if isinstance(patient_data, dict):
        patient_data = pd.DataFrame([patient_data])
    
    # Ensure all features present
    for col in feature_cols:
        if col not in patient_data.columns:
            patient_data[col] = 0
    
    patient_data = patient_data[feature_cols]
    
    # Scale and predict
    patient_scaled = pd.DataFrame(
        scaler.transform(patient_data), 
        columns=feature_cols
    )
    
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0]
    
    # SHAP values
    shap_vals = explainer.shap_values(patient_scaled)
    
    feature_contrib = pd.DataFrame({
        'Feature': feature_cols,
        'SHAP_Value': shap_vals[0],
        'Value': patient_scaled.values[0]
    }).sort_values('SHAP_Value', key=abs, ascending=False)
    
    top_features = feature_contrib.head(5)
    
    return {
        'predicted_class': int(prediction),
        'diagnosis': 'Alzheimer\'s Risk' if prediction == 1 else 'No Risk',
        'risk_score': float(probability[1]),
        'confidence': float(max(probability) * 100),
        'top_features': [
            {
                'feature': row['Feature'],
                'shap_value': float(row['SHAP_Value']),
                'impact': 'Increases Risk' if row['SHAP_Value'] > 0 else 'Decreases Risk'
            }
            for _, row in top_features.iterrows()
        ]
    }

print("\n" + "=" * 80)
print("🧪 TESTING PREDICTION FUNCTION")
print("=" * 80)

# Test on sample patient
sample = X_test.iloc[0:1].copy()
result = predict_alzheimer_risk(sample, model, scaler, explainer, X_train.columns.tolist())

print(f"\n📋 SAMPLE PREDICTION:")
print(f"   Diagnosis: {result['diagnosis']}")
print(f"   Risk Score: {result['risk_score']:.4f} ({result['risk_score']*100:.2f}%)")
print(f"   Confidence: {result['confidence']:.2f}%")
print(f"\n   Top Contributing Features:")
for i, feat in enumerate(result['top_features'], 1):
    print(f"      {i}. {feat['feature']:25s} | {feat['impact']}")

# ============================================================================
# STEP 13: SAVE MODEL
# ============================================================================

print("\n" + "=" * 80)
print("💾 SAVING MODEL AND ARTIFACTS")
print("=" * 80)

# Save model
with open('cogaly_xgb_v1.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Model saved: cogaly_xgb_v1.pkl")

# Save scaler
with open('cogaly_scaler_v1.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved: cogaly_scaler_v1.pkl")

# Save feature columns
with open('cogaly_features_v1.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)
print("✅ Features saved: cogaly_features_v1.pkl")

# Save explainer
with open('cogaly_explainer_v1.pkl', 'wb') as f:
    pickle.dump(explainer, f)
print("✅ Explainer saved: cogaly_explainer_v1.pkl")

# Save metrics
metrics = {
    'accuracy': float(accuracy),
    'roc_auc': float(roc_auc),
    'precision': float(precision),
    'recall': float(recall),
    'n_features': X_train.shape[1]
}

with open('cogaly_metrics_v1.pkl', 'wb') as f:
    pickle.dump(metrics, f)
print("✅ Metrics saved: cogaly_metrics_v1.pkl")

# ============================================================================
# STEP 14: DOWNLOAD FILES (For Google Colab)
# ============================================================================

print("\n" + "=" * 80)
print("📥 DOWNLOAD TRAINED MODEL (Uncomment for Colab)")
print("=" * 80)

# Uncomment to download files in Google Colab:
"""
from google.colab import files
files.download('cogaly_xgb_v1.pkl')
files.download('cogaly_scaler_v1.pkl')
files.download('cogaly_features_v1.pkl')
files.download('cogaly_explainer_v1.pkl')
files.download('cogaly_metrics_v1.pkl')
"""

print("\n✅ To download in Colab, uncomment the download code above")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("🎉 COGALY MODEL TRAINING COMPLETE!")
print("=" * 80)
print(f"""
📊 MODEL SUMMARY:
   • Algorithm: XGBoost Classifier
   • Features: {X_train.shape[1]}
   • Training Samples: {len(X_train):,}
   • Test Samples: {len(X_test):,}

🎯 PERFORMANCE:
   • Accuracy:  {accuracy*100:.2f}%
   • ROC-AUC:   {roc_auc:.4f}
   • Precision: {precision*100:.2f}%
   • Recall:    {recall*100:.2f}%

💾 SAVED FILES:
   • cogaly_xgb_v1.pkl (Main model)
   • cogaly_scaler_v1.pkl (Feature scaler)
   • cogaly_features_v1.pkl (Feature names)
   • cogaly_explainer_v1.pkl (SHAP explainer)
   • cogaly_metrics_v1.pkl (Performance metrics)
""")

print("=" * 80)
print("✅ Ready for production deployment!")
print("=" * 80)

# ============================================================================
# BONUS: LOAD AND TEST SAVED MODEL
# ============================================================================

print("\n" + "=" * 80)
print("🔄 VERIFYING SAVED MODEL")
print("=" * 80)

# Load model
with open('cogaly_xgb_v1.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('cogaly_scaler_v1.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

with open('cogaly_features_v1.pkl', 'rb') as f:
    loaded_features = pickle.load(f)

with open('cogaly_explainer_v1.pkl', 'rb') as f:
    loaded_explainer = pickle.load(f)

print("✅ All models loaded successfully!")

# Test prediction
test_sample = X_test.iloc[10:11]
verify_result = predict_alzheimer_risk(
    test_sample, loaded_model, loaded_scaler, 
    loaded_explainer, loaded_features
)

print(f"\n🧪 VERIFICATION PREDICTION:")
print(f"   Diagnosis: {verify_result['diagnosis']}")
print(f"   Risk Score: {verify_result['risk_score']:.4f}")
print(f"   Confidence: {verify_result['confidence']:.2f}%")

print("\n" + "=" * 80)
print("✅ MODEL VERIFICATION COMPLETE!")
print("=" * 80)
