"""
Cogaly - Early Alzheimer's Risk Detection Model
================================================
A machine learning model using XGBoost for predicting Alzheimer's disease risk
with SHAP explainability for interpretable predictions.

Author: Cogaly Team
Date: December 17, 2025
"""

# ============================================================================
# 1. IMPORTS AND SETUP
# ============================================================================

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

# XGBoost classifier
import xgboost as xgb

# SHAP for model explainability
import shap

# Model persistence
import pickle

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Warnings
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("COGALY - Alzheimer's Risk Detection System")
print("=" * 80)
print("\n✓ All libraries imported successfully\n")


# ============================================================================
# 2. DATA LOADING AND EXPLORATION
# ============================================================================

print("STEP 1: Loading Dataset...")
print("-" * 80)

# Load the dataset
df = pd.read_csv('../data/alzheimers.csv')

print(f"✓ Dataset loaded successfully!")
print(f"  - Total samples: {df.shape[0]:,}")
print(f"  - Total features: {df.shape[1]}")
print(f"\n📊 Dataset Overview:")
print(df.head())
print(f"\n📋 Dataset Info:")
print(df.info())
print(f"\n📈 Target Distribution:")
print(df['Diagnosis'].value_counts())
print(f"\nClass Balance: {df['Diagnosis'].value_counts(normalize=True).to_dict()}")


# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Data Preprocessing...")
print("-" * 80)

# Remove PatientID and DoctorInCharge columns (non-predictive)
columns_to_drop = ['PatientID', 'DoctorInCharge']
df_processed = df.drop(columns=columns_to_drop, errors='ignore')
print(f"✓ Removed non-predictive columns: {columns_to_drop}")

# Separate features and target
X = df_processed.drop('Diagnosis', axis=1)
y = df_processed['Diagnosis']

print(f"✓ Separated features (X) and target (y)")
print(f"  - Feature columns: {X.shape[1]}")
print(f"  - Target column: Diagnosis")

# Check for missing values
missing_values = X.isnull().sum()
if missing_values.sum() > 0:
    print(f"\n⚠️  Missing values detected:")
    print(missing_values[missing_values > 0])
    
    # Handle missing values - fill numeric columns with median
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)
    
    # Handle missing values - fill categorical columns with mode
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].mode()[0], inplace=True)
    
    print(f"✓ Missing values handled")
else:
    print(f"✓ No missing values detected")

# Encode categorical features if any
categorical_columns = X.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    print(f"\n📝 Encoding categorical features: {list(categorical_columns)}")
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    print(f"✓ Categorical features encoded")

print(f"\n✓ Preprocessing complete!")
print(f"  - Final feature count: {X.shape[1]}")
print(f"  - Feature names: {list(X.columns)}")


# ============================================================================
# 4. TRAIN-TEST SPLIT
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Splitting Data into Train and Test Sets...")
print("-" * 80)

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Maintain class distribution
)

print(f"✓ Data split completed (80/20 split)")
print(f"  - Training samples: {X_train.shape[0]:,} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  - Testing samples: {X_test.shape[0]:,} ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"\n  Training set class distribution:")
print(f"    - Class 0 (No Alzheimer's): {(y_train == 0).sum():,}")
print(f"    - Class 1 (Alzheimer's): {(y_train == 1).sum():,}")
print(f"\n  Test set class distribution:")
print(f"    - Class 0 (No Alzheimer's): {(y_test == 0).sum():,}")
print(f"    - Class 1 (Alzheimer's): {(y_test == 1).sum():,}")


# ============================================================================
# 5. FEATURE SCALING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Feature Scaling...")
print("-" * 80)

# Scale features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to maintain feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print(f"✓ Features scaled using StandardScaler")
print(f"  - Mean: 0, Standard Deviation: 1")


# ============================================================================
# 6. MODEL TRAINING - XGBoost Classifier
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Training XGBoost Classifier...")
print("-" * 80)

# Initialize XGBoost classifier with default hyperparameters
model = xgb.XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

print(f"🤖 Model: XGBoost Classifier")
print(f"  - Hyperparameters: Default settings")
print(f"  - Random state: 42")
print(f"\n⏳ Training in progress...")

# Train the model
model.fit(X_train_scaled, y_train)

print(f"✓ Model training completed!")


# ============================================================================
# 7. MODEL PREDICTIONS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Making Predictions...")
print("-" * 80)

# Make predictions on test set
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

print(f"✓ Predictions generated for {len(y_test):,} test samples")


# ============================================================================
# 8. MODEL EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Model Evaluation...")
print("-" * 80)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"📊 MODEL PERFORMANCE METRICS:")
print(f"=" * 80)
print(f"  ✓ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  ✓ ROC-AUC:   {roc_auc:.4f}")
print(f"  ✓ Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  ✓ Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"=" * 80)

# Classification report
print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
print("-" * 80)
print(classification_report(y_test, y_pred, target_names=['No Alzheimer\'s', 'Alzheimer\'s']))

# Confusion matrix
print(f"\n🔢 CONFUSION MATRIX:")
print("-" * 80)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\n  True Negatives:  {cm[0, 0]:,}")
print(f"  False Positives: {cm[0, 1]:,}")
print(f"  False Negatives: {cm[1, 0]:,}")
print(f"  True Positives:  {cm[1, 1]:,}")

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Alzheimer\'s', 'Alzheimer\'s'],
            yticklabels=['No Alzheimer\'s', 'Alzheimer\'s'])
plt.title('Confusion Matrix - Cogaly Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('../model/confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Confusion matrix visualization saved as 'confusion_matrix.png'")


# ============================================================================
# 9. ROC CURVE
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: Generating ROC Curve...")
print("-" * 80)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Cogaly Alzheimer\'s Detection Model')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../model/roc_curve.png', dpi=300, bbox_inches='tight')
print(f"✓ ROC curve visualization saved as 'roc_curve.png'")


# ============================================================================
# 10. SHAP EXPLAINABILITY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: Computing SHAP Values for Explainability...")
print("-" * 80)

print(f"⏳ Initializing SHAP explainer...")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for test set
shap_values = explainer.shap_values(X_test_scaled)

print(f"✓ SHAP values computed for {len(X_test_scaled):,} samples")

# Get feature importance from SHAP
shap_importance = pd.DataFrame({
    'Feature': X_test_scaled.columns,
    'Importance': np.abs(shap_values).mean(axis=0)
}).sort_values('Importance', ascending=False)

print(f"\n🔍 TOP 10 CONTRIBUTING FEATURES (SHAP):")
print("-" * 80)
for idx, row in shap_importance.head(10).iterrows():
    print(f"  {row['Feature']:30s} | Importance: {row['Importance']:.6f}")

# SHAP Summary Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", show=False)
plt.title('SHAP Feature Importance - Cogaly Model')
plt.tight_layout()
plt.savefig('../model/shap_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"\n✓ SHAP feature importance plot saved as 'shap_feature_importance.png'")

# SHAP Detailed Summary Plot
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test_scaled, show=False)
plt.title('SHAP Summary Plot - Feature Impact on Predictions')
plt.tight_layout()
plt.savefig('../model/shap_summary_plot.png', dpi=300, bbox_inches='tight')
print(f"✓ SHAP summary plot saved as 'shap_summary_plot.png'")


# ============================================================================
# 11. PREDICTION FUNCTION WITH EXPLAINABILITY
# ============================================================================

def predict_alzheimer_risk(patient_data, model, scaler, explainer, feature_columns):
    """
    Predict Alzheimer's risk for a single patient with explainability.
    
    Parameters:
    -----------
    patient_data : dict or pd.DataFrame
        Patient features
    model : XGBClassifier
        Trained model
    scaler : StandardScaler
        Fitted scaler
    explainer : shap.TreeExplainer
        SHAP explainer
    feature_columns : list
        List of feature names
    
    Returns:
    --------
    dict : Prediction results with explainability
    """
    
    # Convert to DataFrame if dict
    if isinstance(patient_data, dict):
        patient_data = pd.DataFrame([patient_data])
    
    # Ensure all features are present
    for col in feature_columns:
        if col not in patient_data.columns:
            patient_data[col] = 0  # Fill missing features with 0
    
    # Reorder columns to match training data
    patient_data = patient_data[feature_columns]
    
    # Scale features
    patient_data_scaled = scaler.transform(patient_data)
    patient_data_scaled = pd.DataFrame(patient_data_scaled, columns=feature_columns)
    
    # Make prediction
    prediction = model.predict(patient_data_scaled)[0]
    probability = model.predict_proba(patient_data_scaled)[0]
    
    # Calculate SHAP values for this prediction
    shap_values_patient = explainer.shap_values(patient_data_scaled)
    
    # Get top contributing features
    feature_contributions = pd.DataFrame({
        'Feature': feature_columns,
        'SHAP_Value': shap_values_patient[0],
        'Feature_Value': patient_data_scaled.values[0]
    }).sort_values('SHAP_Value', key=abs, ascending=False)
    
    top_features = feature_contributions.head(5)
    
    # Calculate confidence
    confidence = max(probability) * 100
    
    # Create result dictionary
    result = {
        'predicted_class': int(prediction),
        'diagnosis': 'Alzheimer\'s Risk Detected' if prediction == 1 else 'No Alzheimer\'s Risk',
        'risk_score': float(probability[1]),
        'no_risk_score': float(probability[0]),
        'confidence': float(confidence),
        'top_contributing_features': [
            {
                'feature': row['Feature'],
                'shap_value': float(row['SHAP_Value']),
                'feature_value': float(row['Feature_Value']),
                'impact': 'Increases Risk' if row['SHAP_Value'] > 0 else 'Decreases Risk'
            }
            for _, row in top_features.iterrows()
        ]
    }
    
    return result


print("\n" + "=" * 80)
print("STEP 10: Testing Prediction Function...")
print("-" * 80)

# Test prediction on a sample patient
sample_patient = X_test.iloc[0:1].copy()

prediction_result = predict_alzheimer_risk(
    sample_patient, 
    model, 
    scaler, 
    explainer, 
    X_train.columns.tolist()
)

print(f"🧪 SAMPLE PREDICTION RESULT:")
print(f"=" * 80)
print(f"  Diagnosis: {prediction_result['diagnosis']}")
print(f"  Risk Score: {prediction_result['risk_score']:.4f} ({prediction_result['risk_score']*100:.2f}%)")
print(f"  Confidence: {prediction_result['confidence']:.2f}%")
print(f"\n  Top 5 Contributing Features:")
for i, feature in enumerate(prediction_result['top_contributing_features'], 1):
    print(f"    {i}. {feature['feature']:30s} | SHAP: {feature['shap_value']:8.4f} | {feature['impact']}")
print(f"=" * 80)


# ============================================================================
# 12. SAVE MODEL AND ARTIFACTS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 11: Saving Model and Artifacts...")
print("-" * 80)

# Save the trained model
model_filename = '../model/cogaly_xgb_v1.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"✓ Model saved as '{model_filename}'")

# Save the scaler
scaler_filename = '../model/cogaly_scaler_v1.pkl'
with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)
print(f"✓ Scaler saved as '{scaler_filename}'")

# Save feature columns
feature_columns_filename = '../model/cogaly_feature_columns_v1.pkl'
with open(feature_columns_filename, 'wb') as file:
    pickle.dump(X_train.columns.tolist(), file)
print(f"✓ Feature columns saved as '{feature_columns_filename}'")

# Save SHAP explainer
explainer_filename = '../model/cogaly_shap_explainer_v1.pkl'
with open(explainer_filename, 'wb') as file:
    pickle.dump(explainer, file)
print(f"✓ SHAP explainer saved as '{explainer_filename}'")

# Save model metrics
metrics = {
    'accuracy': float(accuracy),
    'roc_auc': float(roc_auc),
    'precision': float(precision),
    'recall': float(recall),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'n_features': X_train.shape[1],
    'feature_names': X_train.columns.tolist()
}

metrics_filename = '../model/cogaly_metrics_v1.pkl'
with open(metrics_filename, 'wb') as file:
    pickle.dump(metrics, file)
print(f"✓ Model metrics saved as '{metrics_filename}'")


# ============================================================================
# 13. MODEL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETE - COGALY v1.0")
print("=" * 80)
print(f"""
📊 MODEL SUMMARY:
  • Algorithm: XGBoost Classifier
  • Features: {X_train.shape[1]} predictive features
  • Training Samples: {len(X_train):,}
  • Test Samples: {len(X_test):,}
  
🎯 PERFORMANCE:
  • Accuracy:  {accuracy*100:.2f}%
  • ROC-AUC:   {roc_auc:.4f}
  • Precision: {precision*100:.2f}%
  • Recall:    {recall*100:.2f}%
  
💾 SAVED ARTIFACTS:
  • Model: cogaly_xgb_v1.pkl
  • Scaler: cogaly_scaler_v1.pkl
  • Feature Columns: cogaly_feature_columns_v1.pkl
  • SHAP Explainer: cogaly_shap_explainer_v1.pkl
  • Metrics: cogaly_metrics_v1.pkl
  
📈 VISUALIZATIONS:
  • Confusion Matrix: confusion_matrix.png
  • ROC Curve: roc_curve.png
  • SHAP Feature Importance: shap_feature_importance.png
  • SHAP Summary Plot: shap_summary_plot.png
""")

print("=" * 80)
print("✅ Cogaly is ready for Alzheimer's risk detection!")
print("=" * 80)


# ============================================================================
# 14. EXAMPLE: LOAD AND USE SAVED MODEL
# ============================================================================

print("\n" + "=" * 80)
print("BONUS: Example of Loading and Using Saved Model")
print("=" * 80)

# Load saved model
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Load saved scaler
with open(scaler_filename, 'rb') as file:
    loaded_scaler = pickle.load(file)

# Load feature columns
with open(feature_columns_filename, 'rb') as file:
    loaded_features = pickle.load(file)

# Load SHAP explainer
with open(explainer_filename, 'rb') as file:
    loaded_explainer = pickle.load(file)

print(f"✓ Model loaded successfully from disk")
print(f"✓ Ready for production use!")

# Make a prediction with loaded model
test_prediction = predict_alzheimer_risk(
    X_test.iloc[5:6],
    loaded_model,
    loaded_scaler,
    loaded_explainer,
    loaded_features
)

print(f"\n🧪 VERIFICATION PREDICTION:")
print(f"  Diagnosis: {test_prediction['diagnosis']}")
print(f"  Risk Score: {test_prediction['risk_score']:.4f}")
print(f"  Confidence: {test_prediction['confidence']:.2f}%")

print("\n" + "=" * 80)
print("🎉 COGALY MODEL PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)
