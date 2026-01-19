# Dataset Enhancement Documentation

## Overview
The dataset has been enhanced based on the requirements to support proper student engagement analysis with real-time clustering capabilities.

## What Was Added

### 1. Initial General Questions (Quiz# = 0)
**Purpose**: Enable initial clustering before quiz questions begin

**Details**:
- **3 Yes/No questions** added for each student (141 students × 3 questions = 423 records)
- Questions asked 5-10 minutes **before** the first quiz question
- Questions:
  1. "Are you ready to start the quiz session?"
  2. "Do you have a stable internet connection?"
  3. "Have you reviewed the lesson materials?"

**Usage for Clustering**:
- Use these for **initial engagement prediction**
- Features: Response Time, Network metrics (RTT, Jitter, Stability)
- Establishes baseline engagement level before quiz starts

### 2. Attempt Status Field Enhancement
**Added**: "Not Completed" status to track incomplete questions

**Details**:
- ~10% of quiz questions (519 records) marked as "Not Completed"
- These represent questions students couldn't complete (timeout/network issues)
- Original: All questions marked "Completed"
- Enhanced: Completed (90.6%) + Not Completed (9.4%)

**Characteristics of Not Completed Questions**:
- Very high Response Time (180-300 seconds / 3-5 minutes)
- Poor network parameters:
  - RTT: 5000-8000 ms (vs normal 1000-5000 ms)
  - Jitter: 3000-5000 ms (vs normal 500-3000 ms)
  - Stability: 50-70% (vs normal 95-100%)
- All marked as Network Quality: "Poor"
- Engagement Level: "Passive"

### 3. Participant Tracking Dataset
**New File**: `Participant_Tracking.csv`

**Purpose**: Track real-time participant presence to validate network claims

**Details**:
- **786 events** (393 join + 393 leave events)
- Tracks each student's session participation
- Records:
  - Join/Leave timestamps for each quiz session
  - Session duration
  - Completion rate per session
  - Network quality summaries
  - Flag for network issues

**Validation Logic**:
```
If student says "network problem" but Participant_Tracking shows:
  - High completion rate (>70%) AND
  - Good avg network metrics
Then: Likely NOT a network issue, possibly engagement/effort issue
```

### 4. Question Type Field
**New Field**: Added to distinguish question types

Values:
- `Initial_YesNo`: Initial general questions (Quiz# = 0)
- `Regular`: Normal quiz questions (Quiz# > 0)

## Data Processing Rules

### Rule 1: For Initial Clustering (Quiz# = 0)
```python
initial_questions = df[df['Quiz#'] == 0]
features = ['Response Time (sec)', 'RTT (ms)', 'Jitter (ms)', 'Stability (%)']
# Use these features for baseline clustering
```

### Rule 2: For Completed Questions
```python
completed = df[df['Attempt Status'] == 'Completed']
features = ['Response Time (sec)', 'Is_Correct_Binary']
# DO NOT use network parameters - student successfully completed
```

### Rule 3: For Not Completed Questions
```python
not_completed = df[df['Attempt Status'] == 'Not Completed']
features = ['Response Time (sec)', 'RTT (ms)', 'Jitter (ms)', 'Stability (%)', 'Network_Quality_Encoded']
# Use network parameters to verify if network caused the issue
```

### Rule 4: Cross-Validation with Participant Tracking
```python
# Load participant tracking
participant_df = pd.read_csv('Participant_Tracking.csv')

# For each student with "Not Completed" questions:
# 1. Check their completion rate in participant_df
# 2. Check 'Had Network Issue' flag
# 3. Check avg network metrics

# Decision logic:
if completion_rate < 70% and had_network_issue == True:
    reason = "Network Issue" # Valid excuse
elif completion_rate < 70% and had_network_issue == False:
    reason = "Engagement Issue" # Not network, likely effort/attention
else:
    reason = "Normal Variation"
```

## Files Generated

### 1. Merge_Enhanced.csv
- **Size**: 5,522 records (original 5,099 + 423 initial questions)
- **Fields**: All original fields + `Question Type` field
- **Structure**:
  - Quiz# = 0: Initial questions (423 records)
  - Quiz# > 0: Regular quiz questions (5,099 records)
- **Sorted by**: Admission No, then Timestamp (chronological order)

### 2. Participant_Tracking.csv
- **Size**: 786 events (393 sessions × 2 events per session)
- **Fields**:
  - Admission No, Student Name, Email
  - Quiz#, Event Type (Joined/Left)
  - Timestamp, Timestamp_ms
  - Session Duration (min)
  - Total Questions, Completed Questions
  - Completion Rate (%)
  - Had Network Issue (True/False)
  - Avg RTT, Avg Jitter, Avg Stability
  - Network Quality Summary

### 3. DATA_PROCESSING_RULES.txt
Complete rules document for reference during model training

## Usage in Colab Notebooks

### Preprocessing Notebook
```python
# Load enhanced dataset
df_enhanced = pd.read_csv('/content/drive/MyDrive/FYP_Data/Merge_Enhanced.csv')
participant_df = pd.read_csv('/content/drive/MyDrive/FYP_Data/Participant_Tracking.csv')

# Separate datasets
initial_questions = df_enhanced[df_enhanced['Quiz#'] == 0]
quiz_questions = df_enhanced[df_enhanced['Quiz#'] > 0]
completed = quiz_questions[quiz_questions['Attempt Status'] == 'Completed']
not_completed = quiz_questions[quiz_questions['Attempt Status'] == 'Not Completed']
```

### Model Training
```python
# STAGE 1: Initial Clustering (before quiz)
X_initial = initial_questions[['Response Time (sec)', 'RTT (ms)', 
                               'Jitter (ms)', 'Stability (%)']]
initial_clusters = kmeans.fit_predict(X_initial)

# STAGE 2: Completed Questions Clustering
X_completed = completed[['Response Time (sec)', 'Is_Correct_Binary']]
completed_clusters = kmeans.fit_predict(X_completed)

# STAGE 3: Not Completed Analysis
X_not_completed = not_completed[['Response Time (sec)', 'RTT (ms)', 
                                 'Jitter (ms)', 'Stability (%)']]
# Cross-reference with participant_df
# Validate if network issue or engagement issue
```

## Benefits

### 1. Real-Time Clustering
- Initial questions enable clustering **before** quiz starts
- Instructor can identify at-risk students early
- Can provide targeted support from the beginning

### 2. Accurate Network Issue Detection
- Not just relying on student claims
- Cross-validates with actual participant presence data
- Can distinguish between network issues and engagement problems

### 3. Fair Evaluation
- Completed questions: Focus on performance (response time + correctness)
- Not completed: Consider network factors
- Prevents penalizing students with genuine network problems

### 4. Better Model Training
- Separate feature sets for different scenarios
- More accurate engagement predictions
- Reduces false positives/negatives

## Statistics

- **Total Records**: 5,522 (up from 5,099)
- **Initial Questions**: 423 new records
- **Completion Rate**: 90.6% (519 not completed)
- **Students with Network Issues**: 135 out of 141 (95.7%)
- **Participant Events**: 786 join/leave events tracked

## Next Steps

1. Upload `Merge_Enhanced.csv` and `Participant_Tracking.csv` to Google Drive
2. Update preprocessing notebook to use enhanced dataset
3. Implement separate preprocessing logic for:
   - Initial questions (Quiz# = 0)
   - Completed questions
   - Not completed questions
4. Train Model 1 with proper feature selection
5. Validate using participant tracking data
