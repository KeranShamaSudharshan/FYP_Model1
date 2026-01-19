# Notebooks and Datasets Guide

This directory contains Google Colab notebooks for preprocessing and training the student engagement models.

## 📁 Datasets Required

### 1. Merge_Enhanced.csv (5,522 records)
**Location**: Upload to `/content/drive/MyDrive/FYP_Data/`

**Description**: Enhanced dataset with initial questions and completion tracking

**Structure**:
- **Quiz# = 0**: Initial Yes/No questions (423 records, 3 per student)
- **Quiz# > 0**: Regular quiz questions (5,099 records)
- **Attempt Status**: "Completed" or "Not Completed"
- **Question Type**: "Initial_YesNo" or "Regular"

**Key Fields**:
```
Quiz#, Student Name, Admission No, Email, Class, Question ID, Question,
Selected Answer, Correct Answer, Is Correct, Response Time (ms),
Response Time (sec), Engagement Level, Attempt Status, Answered At,
Timestamp, RTT (ms), Jitter (ms), Stability (%), Network Quality, Question Type
```

**Download from**: Root directory of this repository

### 2. Participant_Tracking.csv (786 events)
**Location**: Upload to `/content/drive/MyDrive/FYP_Data/`

**Description**: Real-time join/leave tracking for validation

**Structure**:
- Join/Leave events for each quiz session
- Completion rates and network issue flags
- Used to validate if students claiming network issues actually had them

**Key Fields**:
```
Admission No, Student Name, Email, Quiz#, Event Type (Joined/Left),
Timestamp, Session Duration (min), Total Questions, Completed Questions,
Completion Rate (%), Had Network Issue, Avg RTT (ms), Avg Jitter (ms),
Avg Stability (%), Network Quality Summary
```

**Download from**: Root directory of this repository

## 📓 Notebooks

### 1. 01_Preprocessing_Enhanced_Dataset.ipynb ✅
**Purpose**: Load and preprocess the enhanced dataset

**Key Features**:
- ✅ Filters only participating students (students who joined sessions)
- ✅ Separates initial questions (Quiz# = 0) from quiz questions
- ✅ Applies different feature selection:
  - **Initial questions**: Response Time + All network metrics
  - **Completed questions**: Response Time + Is Correct only (NO network params)
  - **Not Completed**: Response Time + Network metrics
- ✅ Cross-validates with participant tracking
- ✅ Scales and saves preprocessed data

**Outputs Saved to** `/content/drive/MyDrive/FYP_Data/Preprocessed/`:
- `X_initial_scaled.npy`, `y_initial.npy`
- `X_completed_scaled.npy`, `y_completed.npy`
- `X_not_completed_scaled.npy`, `y_not_completed.npy`
- `scaler_initial.pkl`, `scaler_completed.pkl`, `scaler_not_completed.pkl`

**Run Time**: ~2-3 minutes

### 2. 02_Model_Training.ipynb (To be created)
**Purpose**: Train clustering and prediction models

**Models to Include**:
1. **K-Means Clustering**
   - 3 clusters (Passive, Moderate, Active)
   - Elbow method validation
   - Silhouette Score, Davies-Bouldin Index
   
2. **Random Forest Classifier**
   - Handle class imbalance with SMOTE or class_weight
   - Feature importance analysis
   - Cross-validation
   
3. **XGBoost Classifier**
   - Advanced gradient boosting
   - Better handling of imbalanced data
   - Hyperparameter tuning

**Evaluation Metrics**:
- Clustering: Silhouette, Davies-Bouldin, Calinski-Harabasz
- Classification: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion Matrix
- Classification Report

## 🚀 Quick Start Guide

### Step 1: Upload Datasets to Google Drive
```python
# In Google Colab
from google.colab import drive
drive.mount('/content/drive')

# Create directory
!mkdir -p /content/drive/MyDrive/FYP_Data

# Upload Merge_Enhanced.csv and Participant_Tracking.csv to this folder
```

### Step 2: Run Preprocessing Notebook
1. Open `01_Preprocessing_Enhanced_Dataset.ipynb` in Google Colab
2. Run all cells
3. Verify outputs are saved to `/content/drive/MyDrive/FYP_Data/Preprocessed/`

### Step 3: Run Model Training Notebook
1. Open `02_Model_Training.ipynb` in Google Colab
2. Run all cells
3. Review evaluation metrics
4. Download trained models

## 📊 Expected Results

### Preprocessing
- **Initial Questions**: ~423 records from 141 students
- **Completed Questions**: ~4,580 records (90.6%)
- **Not Completed**: ~519 records (9.4%)
- **Participating Students**: 141 (100% participation rate)

### Clustering Model
- **Algorithm**: K-Means (k=3)
- **Expected Silhouette Score**: 0.45-0.50 (Good)
- **Expected Davies-Bouldin**: 0.80-0.90 (Excellent)
- **Clusters**: Passive (37%), Moderate (56%), Active (7%)

### Prediction Model
- **Algorithm**: Random Forest / XGBoost
- **Expected Accuracy**: 75-85%
- **Expected F1-Score**: 0.70-0.80
- **Challenge**: Class imbalance (Moderate only 5.5%)

## 🔧 Data Processing Rules

### Rule 1: Filter Non-Participants
```python
# Get students who joined sessions
participated = participant_df[
    participant_df['Event Type'] == 'Joined'
]['Admission No'].unique()

# Filter dataset
df = df[df['Admission No'].isin(participated)]
```

### Rule 2: Feature Selection by Status

**For Completed Questions**:
```python
features = ['Response Time (sec)', 'Is_Correct_Binary']
# NO network parameters
```

**For Not Completed Questions**:
```python
features = ['Response Time (sec)', 'RTT (ms)', 'Jitter (ms)', 
            'Stability (%)', 'Network_Quality_Encoded']
# Include network parameters for validation
```

**For Initial Questions**:
```python
features = ['Response Time (sec)', 'RTT (ms)', 'Jitter (ms)', 'Stability (%)']
# All network parameters for baseline
```

### Rule 3: Cross-Validation Logic
```python
# Validate network claims
if completion_rate < 70% and had_network_issue == True:
    reason = "Valid Network Issue"
elif completion_rate < 70% and had_network_issue == False:
    reason = "Engagement Issue (not network)"
else:
    reason = "Normal Performance"
```

## 📚 Documentation Files

Additional documentation in the root directory:
- `DATA_PROCESSING_RULES.txt` - Complete processing guidelines
- `DATASET_ENHANCEMENT_DOCS.md` - Dataset enhancement details
- `CHANGES_SUMMARY.md` - Summary of all changes
- `enhance_dataset.py` - Script to regenerate enhanced dataset

## 🐛 Troubleshooting

### Issue: "File not found"
**Solution**: Verify dataset paths match:
```python
DATA_PATH = '/content/drive/MyDrive/FYP_Data/'
```

### Issue: "All students filtered out"
**Solution**: Check that Participant_Tracking.csv is loaded correctly

### Issue: "Class imbalance warnings"
**Solution**: This is expected. The notebooks handle it with SMOTE or class_weight

### Issue: "Low clustering scores"
**Solution**: Expected for real-world data. Scores of 0.45-0.50 are acceptable

## 📞 Support

For questions about:
- **Dataset structure**: See `DATASET_ENHANCEMENT_DOCS.md`
- **Processing rules**: See `DATA_PROCESSING_RULES.txt`
- **Implementation**: See `IMPLEMENTATION_SUMMARY.md`

## ✅ Checklist

Before running notebooks:
- [ ] Uploaded `Merge_Enhanced.csv` to Google Drive
- [ ] Uploaded `Participant_Tracking.csv` to Google Drive
- [ ] Created `/content/drive/MyDrive/FYP_Data/` directory
- [ ] Mounted Google Drive in Colab
- [ ] Verified file paths are correct

After preprocessing:
- [ ] Verified preprocessed files are saved
- [ ] Checked participant filtering worked
- [ ] Reviewed feature shapes
- [ ] Confirmed scalers are saved

After model training:
- [ ] Reviewed clustering metrics
- [ ] Checked prediction accuracy
- [ ] Examined confusion matrix
- [ ] Saved trained models

## 🎯 Summary

**Key Points**:
1. ✅ Only participating students are clustered (non-participants excluded)
2. ✅ Completed questions use Response Time + Correctness (NO network params)
3. ✅ Not Completed questions use Response Time + Network params for validation
4. ✅ Initial questions establish baseline engagement
5. ✅ Participant tracking validates network issue claims

**Ready to use!** Start with `01_Preprocessing_Enhanced_Dataset.ipynb`
