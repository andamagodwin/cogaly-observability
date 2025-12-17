# Model Card: Cogaly v1.0

## Model Details

**Model Name**: Cogaly - Alzheimer's Risk Detection Model  
**Version**: 1.0  
**Date**: December 17, 2025  
**Model Type**: XGBoost Classifier (Gradient Boosting)  
**Framework**: XGBoost 1.7.0+, Scikit-learn 1.2.0+  
**License**: MIT  

## Intended Use

### Primary Intended Uses
- Early detection of Alzheimer's disease risk in adult patients
- Clinical decision support for healthcare professionals
- Research and analysis of Alzheimer's risk factors

### Primary Intended Users
- Healthcare professionals (doctors, neurologists, geriatricians)
- Clinical researchers
- Healthcare institutions and hospitals

### Out-of-Scope Use Cases
- **Not for self-diagnosis** by patients
- **Not a replacement** for professional medical diagnosis
- **Not for use** on children or adolescents
- **Not validated** for other forms of dementia

## Model Architecture

### Algorithm
XGBoost (eXtreme Gradient Boosting) Classifier with tree-based ensemble learning

### Input Features (32 total)

**Demographics:**
- Age, Gender, Ethnicity, EducationLevel

**Lifestyle Factors:**
- BMI, Smoking, AlcoholConsumption
- PhysicalActivity, DietQuality, SleepQuality

**Medical History:**
- FamilyHistoryAlzheimers, CardiovascularDisease
- Diabetes, Depression, HeadInjury, Hypertension

**Clinical Measurements:**
- SystolicBP, DiastolicBP
- CholesterolTotal, CholesterolLDL, CholesterolHDL, CholesterolTriglycerides

**Cognitive & Functional Assessments:**
- MMSE (Mini-Mental State Examination)
- FunctionalAssessment
- ADL (Activities of Daily Living)

**Behavioral Indicators:**
- MemoryComplaints, BehavioralProblems
- Confusion, Disorientation, PersonalityChanges
- DifficultyCompletingTasks, Forgetfulness

### Output
- **Binary Classification**: 0 (No Alzheimer's Risk) or 1 (Alzheimer's Risk)
- **Risk Score**: Probability [0-1] of Alzheimer's disease
- **Confidence**: Percentage confidence in prediction
- **Feature Contributions**: Top 5 features with SHAP values

## Training Data

### Dataset
- **Source**: Synthetic Alzheimer's patient data (alzheimers.csv)
- **Size**: 2,149 patient records
- **Features**: 35 columns (32 used for training after preprocessing)
- **Target Distribution**:
  - Class 0 (No Alzheimer's): 1,389 samples (64.6%)
  - Class 1 (Alzheimer's): 760 samples (35.4%)

### Data Split
- **Training Set**: 1,719 samples (80%)
- **Test Set**: 430 samples (20%)
- **Stratified Split**: Maintains class distribution

### Preprocessing
1. Removed non-predictive columns (PatientID, DoctorInCharge)
2. Checked for missing values (none found)
3. Encoded categorical features (if any)
4. Standardized features using StandardScaler (mean=0, std=1)

## Performance Metrics

### Test Set Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 94.42% |
| **ROC-AUC** | 0.9480 |
| **Precision** | 93.84% |
| **Recall** | 90.13% |
| **F1-Score** | 0.92 (Alzheimer's class) |

### Confusion Matrix

|                    | Predicted: No Risk | Predicted: Risk |
|--------------------|--------------------|-----------------|
| **Actual: No Risk** | 269 (TN) | 9 (FP) |
| **Actual: Risk** | 15 (FN) | 137 (TP) |

### Interpretation
- **True Negatives**: 269 correctly identified no-risk patients
- **True Positives**: 137 correctly identified at-risk patients
- **False Positives**: 9 healthy patients incorrectly flagged (3.2%)
- **False Negatives**: 15 at-risk patients missed (9.9%)

### Clinical Implications
- **High Recall (90.13%)**: Catches most at-risk patients (important for early detection)
- **High Precision (93.84%)**: Minimizes false alarms
- **Low False Negative Rate**: Only 15 at-risk patients missed out of 152

## Explainability

### SHAP (SHapley Additive exPlanations)

The model uses SHAP values to explain predictions:

**Top 5 Most Important Features:**
1. **FunctionalAssessment** (Importance: 2.32)
2. **ADL** - Activities of Daily Living (Importance: 2.06)
3. **MemoryComplaints** (Importance: 1.57)
4. **MMSE** - Cognitive Test Score (Importance: 1.41)
5. **BehavioralProblems** (Importance: 1.34)

### Feature Impact
- **Positive SHAP values**: Increase risk of Alzheimer's
- **Negative SHAP values**: Decrease risk of Alzheimer's
- Higher absolute values = stronger impact on prediction

## Ethical Considerations

### Fairness
- Model trained on diverse demographics (multiple ethnicities)
- Performance should be monitored across demographic groups
- Potential for algorithmic bias if training data not representative

### Privacy
- Patient data must be handled according to HIPAA/GDPR
- Model does not store patient identifiers
- Predictions should be kept confidential

### Limitations
- **Not a diagnosis**: Model provides risk assessment, not definitive diagnosis
- **Requires validation**: Clinical validation needed before deployment
- **Dataset limitations**: Training data may not represent all populations
- **Temporal validity**: May require retraining as medical knowledge evolves

## Limitations and Risks

### Known Limitations
1. **Training data size**: 2,149 samples (larger datasets may improve performance)
2. **Synthetic data**: Trained on synthetic data; real-world validation needed
3. **Feature completeness**: Missing features (e.g., genetic markers, brain imaging)
4. **Class imbalance**: 64% no risk vs 36% risk (addressed with stratification)
5. **Temporal factors**: Single time-point data (no longitudinal tracking)

### Potential Risks
- **False Negatives**: May miss 10% of at-risk patients
- **False Positives**: May cause unnecessary anxiety in 3% of healthy patients
- **Overreliance**: Should not replace comprehensive medical evaluation
- **Population drift**: Performance may degrade if patient demographics change

## Model Maintenance

### Monitoring
- Track performance metrics on new data
- Monitor for data drift and concept drift
- Regular audits for fairness across demographics

### Retraining
- Retrain when performance degrades
- Update with new medical research findings
- Incorporate additional features as available

### Versioning
- Current version: v1.0
- Version control for reproducibility
- Document all changes in model updates

## Technical Specifications

### Hyperparameters
- Default XGBoost parameters
- Random state: 42 (for reproducibility)
- Evaluation metric: Log loss

### Computational Requirements
- **Training time**: ~5 seconds on standard CPU
- **Inference time**: <1ms per prediction
- **Memory**: ~50MB for model artifacts
- **Dependencies**: See requirements.txt

### Model Files
- `cogaly_xgb_v1.pkl` (1.2 MB) - Trained model
- `cogaly_scaler_v1.pkl` (3 KB) - Feature scaler
- `cogaly_feature_columns_v1.pkl` (2 KB) - Feature names
- `cogaly_shap_explainer_v1.pkl` (1.5 MB) - Explainer
- `cogaly_metrics_v1.pkl` (1 KB) - Performance metrics

## Validation and Testing

### Cross-Validation
- Train-test split (80/20) with stratification
- Random state ensures reproducibility
- Test set never seen during training

### Performance Stability
- Consistent performance across test set
- High confidence predictions (average >90%)
- Robust to feature scaling

## Deployment Considerations

### Production Readiness
✅ Model serialized and ready for deployment  
✅ Feature preprocessing automated  
✅ Explainability built-in (SHAP)  
✅ Performance metrics documented  
⚠️ Requires clinical validation  
⚠️ Needs regulatory approval for medical use  

### Integration Requirements
1. Input validation for all 32 features
2. Preprocessing pipeline (StandardScaler)
3. Prediction endpoint with SHAP explanations
4. Monitoring and logging infrastructure
5. Error handling and fallback mechanisms

## References

### Frameworks
- XGBoost: Chen & Guestrin (2016)
- SHAP: Lundberg & Lee (2017)
- Scikit-learn: Pedregosa et al. (2011)

### Medical Context
- Alzheimer's Association Guidelines
- MMSE: Folstein et al. (1975)
- ADL Assessment: Katz et al. (1963)

## Contact

For questions about this model:
- Technical issues: GitHub Issues
- Clinical questions: Consult medical professionals
- Collaboration: Contact repository maintainers

## Changelog

### Version 1.0 (December 17, 2025)
- Initial release
- XGBoost classifier with default parameters
- 32 input features
- 94.42% accuracy on test set
- SHAP explainability integrated

---

**Disclaimer**: This model is intended for research and educational purposes. It has not been approved by regulatory agencies for clinical diagnosis. Always consult qualified healthcare professionals for medical decisions.
