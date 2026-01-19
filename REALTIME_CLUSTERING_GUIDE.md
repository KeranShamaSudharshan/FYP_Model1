# Real-Time Clustering System Guide

## Overview

This system implements **real-time student engagement clustering** using a supervised machine learning approach. The model predicts and updates student clusters dynamically during live sessions based on their performance.

## Key Concept

**No Pre-Computed History**: Instead of tracking historical cluster changes in a CSV, we train ML models that can **predict clusters in real-time** based on current student performance metrics.

## System Workflow

### Phase 1: Preprocessing (Offline)
Creates training data from historical sessions with dynamic cluster labels.

```
Input: Merge_Enhanced_Fixed.csv, Participant_Tracking.csv
Process:
  1. Filter participating students only
  2. Separate initial questions from regular questions
  3. For each student, for each question:
     - Calculate cumulative metrics (accuracy, avg response time)
     - Apply cluster assignment rules
     - Generate label: Passive/Moderate/Active
  4. Create feature matrix + label vector
Output: Final_Training_Data.csv, preprocessed arrays
```

### Phase 2: Model Training (Offline)
Trains two models for different clustering stages.

```
Model 1 - Initial Clustering (K-Means):
  Input: Initial question features (response time, network metrics)
  Output: Initial cluster assignment
  Purpose: Baseline classification when student first joins

Model 2 - Dynamic Re-Clustering (Random Forest):
  Input: Cumulative features (accuracy, avg response, question count)
  Output: Updated cluster assignment
  Purpose: Real-time updates as student answers questions
  
Saves: kmeans_initial.pkl, rf_dynamic.pkl
```

### Phase 3: Real-Time Inference (Production)
Deployed models predict clusters on-the-fly during live sessions.

```
Student joins session:
  ├─ Step 1: Answer initial question
  │   ├─ Extract features (response time, network quality)
  │   ├─ Model 1 (K-Means) predicts: "Moderate"
  │   └─ Instructor sees: Student classified as Moderate
  │
  ├─ Step 2: Answer Question 1
  │   ├─ Update cumulative metrics (1 answered, 100% accuracy)
  │   ├─ Model 2 (Random Forest) predicts: "Active"
  │   └─ Cluster updated: Moderate → Active
  │
  ├─ Step 3: Answer Question 2
  │   ├─ Update cumulative metrics (2 answered, 50% accuracy)
  │   ├─ Model 2 predicts: "Moderate"
  │   └─ Cluster updated: Active → Moderate
  │
  └─ Process continues for each question...
```

## Dynamic Cluster Assignment Rules

During preprocessing, clusters are assigned based on cumulative performance:

```python
def assign_cluster(accuracy, avg_response_time, has_network_issue):
    """
    Rules for cluster assignment (used to create training labels)
    """
    if has_network_issue:
        # Network issue prevents accurate classification
        return 'Passive'
    
    if accuracy > 0.80 and avg_response_time < 30:
        return 'Active'      # High performance
    elif accuracy > 0.50 and avg_response_time < 60:
        return 'Moderate'    # Medium performance
    else:
        return 'Passive'     # Needs attention
```

## Feature Engineering

### Initial Question Features
Used for Model 1 (K-Means):
- Response Time (seconds)
- RTT (Round Trip Time)
- Jitter (network variability)
- Stability (network connection quality)

### Dynamic Question Features
Used for Model 2 (Random Forest):
- Cumulative Accuracy (correct / total questions)
- Average Response Time (mean across all questions)
- Total Questions Answered
- Current Response Time
- Is Correct (current question)
- RTT, Jitter, Stability (if question not completed)

## Participant Filtering

**Critical Rule**: Only students who **joined the session** are included in clustering.

```python
# Check participation
participant_df = pd.read_csv('Participant_Tracking.csv')
participated = participant_df[
    participant_df['Event Type'] == 'Joined'
]['Admission No'].unique()

# Filter dataset
df = df[df['Admission No'].isin(participated)]
```

## Implementation Files

### Notebooks

1. **01_Preprocessing_Final.ipynb**
   - Loads raw data
   - Filters participants
   - Applies dynamic cluster logic
   - Creates training labels
   - Saves preprocessed data

2. **02_Model_Training_RealTime.ipynb**
   - Trains K-Means (initial clustering)
   - Trains Random Forest (dynamic updates)
   - Evaluates models
   - Saves trained models

3. **03_RealTime_Inference_Demo.ipynb**
   - Simulates real-time session
   - Shows cluster updates
   - Demonstrates production usage

### Data Files

- **Merge_Enhanced_Fixed.csv**: Raw dataset (5,240 records)
- **Participant_Tracking.csv**: Join/leave events (786 events)
- **Final_Training_Data.csv**: Preprocessed training data with labels

### Model Files

- **kmeans_initial.pkl**: K-Means model for initial clustering
- **rf_dynamic.pkl**: Random Forest for dynamic updates
- **scaler.pkl**: Feature scaler

## Usage Examples

### Training (Offline)

```python
# Step 1: Preprocessing
from notebooks import preprocessing
X_train, y_train = preprocessing.prepare_training_data(
    'Merge_Enhanced_Fixed.csv',
    'Participant_Tracking.csv'
)

# Step 2: Train models
from notebooks import training
kmeans_model, rf_model = training.train_models(X_train, y_train)

# Step 3: Save models
joblib.dump(kmeans_model, 'kmeans_initial.pkl')
joblib.dump(rf_model, 'rf_dynamic.pkl')
```

### Inference (Real-Time)

```python
import joblib
import numpy as np

# Load trained models
kmeans = joblib.load('kmeans_initial.pkl')
rf_model = joblib.load('rf_dynamic.pkl')
scaler = joblib.load('scaler.pkl')

# Student joins and answers initial question
initial_features = [45, 120, 50, 0.85]  # response_time, rtt, jitter, stability
initial_features_scaled = scaler.transform([initial_features])
initial_cluster = kmeans.predict(initial_features_scaled)[0]
print(f"Initial Cluster: {initial_cluster}")  # → "Moderate"

# Student answers first question
cumulative_features = [1.0, 25, 1, 25, 1]  # accuracy, avg_time, count, curr_time, is_correct
cumulative_features_scaled = scaler.transform([cumulative_features])
updated_cluster = rf_model.predict(cumulative_features_scaled)[0]
print(f"Updated Cluster: {updated_cluster}")  # → "Active"

# Student answers second question (incorrectly)
cumulative_features = [0.5, 30, 2, 35, 0]  # 50% accuracy now
cumulative_features_scaled = scaler.transform([cumulative_features])
updated_cluster = rf_model.predict(cumulative_features_scaled)[0]
print(f"Updated Cluster: {updated_cluster}")  # → "Moderate"
```

## Model Performance

### Expected Metrics

**K-Means (Initial Clustering)**:
- Silhouette Score: 0.45-0.50 (good separation)
- Davies-Bouldin Index: 0.80-0.90 (excellent)
- Purpose: Quick baseline classification

**Random Forest (Dynamic Updates)**:
- Accuracy: 80-85%
- Precision: 0.78-0.83
- Recall: 0.75-0.82
- F1-Score: 0.76-0.82
- Purpose: Accurate real-time updates

## Key Advantages

1. **No History Tracking**: Models predict directly from features
2. **Real-Time Updates**: Instant cluster prediction
3. **Scalable**: Works for any number of students
4. **Supervised Learning**: Learns from labeled training data
5. **Production-Ready**: Trained models can be deployed
6. **Fair Classification**: Handles network issues appropriately

## Integration with Production System

```python
class RealTimeClustering:
    def __init__(self):
        self.kmeans = joblib.load('kmeans_initial.pkl')
        self.rf_model = joblib.load('rf_dynamic.pkl')
        self.scaler = joblib.load('scaler.pkl')
        self.student_state = {}  # Track cumulative metrics
    
    def initial_cluster(self, student_id, response_time, rtt, jitter, stability):
        """Predict initial cluster from first question"""
        features = np.array([[response_time, rtt, jitter, stability]])
        features_scaled = self.scaler.transform(features)
        cluster = self.kmeans.predict(features_scaled)[0]
        
        # Initialize tracking
        self.student_state[student_id] = {
            'correct': 0,
            'total': 0,
            'response_times': [],
            'cluster': cluster
        }
        return cluster
    
    def update_cluster(self, student_id, response_time, is_correct):
        """Update cluster after each question"""
        state = self.student_state[student_id]
        
        # Update cumulative metrics
        state['correct'] += int(is_correct)
        state['total'] += 1
        state['response_times'].append(response_time)
        
        # Calculate features
        accuracy = state['correct'] / state['total']
        avg_time = np.mean(state['response_times'])
        
        features = np.array([[
            accuracy,
            avg_time,
            state['total'],
            response_time,
            int(is_correct)
        ]])
        
        features_scaled = self.scaler.transform(features)
        new_cluster = self.rf_model.predict(features_scaled)[0]
        
        # Update state
        old_cluster = state['cluster']
        state['cluster'] = new_cluster
        
        return new_cluster, (old_cluster != new_cluster)
```

## Alignment with User Stories

- **US #38**: Initial question triggers K-Means clustering
- **US #39**: Students clustered based on behavior + performance
- **US #40**: Clusters update in real-time using Random Forest
- **US #41**: Instructors view current cluster assignments
- **US #42**: System stores model predictions (in database, not CSV)

## Conclusion

This approach provides a **production-ready, scalable, and accurate** real-time clustering system that can be deployed in live video conferencing sessions. Models are trained offline and make instant predictions during sessions, with no need for pre-computed cluster histories.
