# Datasets Quick Reference

## Files Available for Download

### 1. Merge_Enhanced.csv (2.9 MB, 5,522 records)
**Purpose**: Main dataset with initial questions and completion status

**Size**: 5,522 records × 21 columns

**Students**: 141 unique students

**Structure**:
- **423 Initial Questions** (Quiz# = 0): 3 Yes/No questions per student for baseline clustering
- **5,099 Quiz Questions** (Quiz# = 4, 5, 6): Regular physics quiz questions

**Completion Breakdown**:
- Completed: 5,003 records (90.6%)
- Not Completed: 519 records (9.4%)

**Columns**:
```
1.  Quiz# - Quiz session number (0 = initial questions, 4/5/6 = regular quizzes)
2.  Student Name - Student full name
3.  Admission No - Unique student ID (20001-20244)
4.  Email - Student email
5.  Class - Subject (Physics)
6.  Question ID - Unique question identifier
7.  Question - Question text (in Tamil)
8.  Selected Answer - Student's answer (0-4 for multiple choice, Yes/No for initial)
9.  Correct Answer - Correct answer index
10. Is Correct - "Yes", "No", or "N/A" (for initial questions)
11. Response Time (ms) - Time taken in milliseconds
12. Response Time (sec) - Time taken in seconds
13. Engagement Level - "Active", "Moderate", or "Passive"
14. Attempt Status - "Completed" or "Not Completed"
15. Answered At - ISO timestamp when answered
16. Timestamp - Unix timestamp (milliseconds)
17. RTT (ms) - Round Trip Time (network metric)
18. Jitter (ms) - Network jitter
19. Stability (%) - Network stability percentage
20. Network Quality - "Poor", "Fair", "Good", or "Excellent"
21. Question Type - "Initial_YesNo" or "Regular"
```

**Usage**:
```python
import pandas as pd
df = pd.read_csv('Merge_Enhanced.csv')

# Separate by question type
initial_q = df[df['Quiz#'] == 0]  # 423 records
regular_q = df[df['Quiz#'] > 0]   # 5,099 records

# Separate by completion status
completed = df[df['Attempt Status'] == 'Completed']     # 5,003 records
not_completed = df[df['Attempt Status'] == 'Not Completed']  # 519 records
```

**Important Notes**:
- Same Admission No = Same Student across all questions
- Records are sorted chronologically per student (by Timestamp)
- Initial questions are 5-10 minutes before quiz questions
- Not Completed questions have high response times (3-5 min) and poor network

---

### 2. Participant_Tracking.csv (124 KB, 786 events)
**Purpose**: Real-time join/leave tracking for session validation

**Size**: 786 records × 16 columns

**Structure**: 393 Join events + 393 Leave events (one pair per quiz session per student)

**Sessions Tracked**: Quiz# 4, 5, 6

**Columns**:
```
1.  Admission No - Student ID
2.  Student Name - Student name
3.  Email - Student email
4.  Quiz# - Quiz session number
5.  Event Type - "Joined" or "Left"
6.  Timestamp - ISO timestamp of event
7.  Timestamp_ms - Unix timestamp (milliseconds)
8.  Session Duration (min) - Duration of session (only for "Left" events)
9.  Total Questions - Total questions in that quiz
10. Completed Questions - Number of questions completed
11. Completion Rate (%) - Percentage of questions completed
12. Had Network Issue - True/False flag
13. Avg RTT (ms) - Average RTT for that session
14. Avg Jitter (ms) - Average jitter
15. Avg Stability (%) - Average stability
16. Network Quality Summary - Most common network quality
```

**Usage**:
```python
import pandas as pd
participant_df = pd.read_csv('Participant_Tracking.csv')

# Get students who joined
joined = participant_df[participant_df['Event Type'] == 'Joined']
students = joined['Admission No'].unique()  # 141 students

# Check completion rates
avg_completion = joined['Completion Rate (%)'].mean()  # Should be ~90%

# Identify students with network issues
network_issues = joined[joined['Had Network Issue'] == True]
```

**Cross-Validation Logic**:
```python
# Validate if "not completed" was due to network
if completion_rate < 70 and had_network_issue == True:
    # Valid excuse - genuine network problem
    use_network_features = True
elif completion_rate < 70 and had_network_issue == False:
    # Likely engagement issue, not network
    use_network_features = False
```

**Important Notes**:
- All 141 students have join/leave events (100% participation)
- 135 students (95.7%) experienced network issues at some point
- Session duration can be negative (timestamp inconsistency) - ignore for analysis
- Use 'Joined' events for most analyses (contains all metrics)

---

## How to Use Together

### Step 1: Filter Participating Students
```python
# Load both datasets
df = pd.read_csv('Merge_Enhanced.csv')
participant_df = pd.read_csv('Participant_Tracking.csv')

# Get students who participated
participated = participant_df[
    participant_df['Event Type'] == 'Joined'
]['Admission No'].unique()

# Filter main dataset
df_filtered = df[df['Admission No'].isin(participated)]
```

### Step 2: Validate Not Completed Questions
```python
# Get not completed questions
not_completed = df_filtered[df_filtered['Attempt Status'] == 'Not Completed']

# Merge with participant data
not_completed_validated = not_completed.merge(
    participant_df[participant_df['Event Type'] == 'Joined'],
    on=['Admission No', 'Quiz#'],
    how='left'
)

# Categorize
not_completed_validated['Issue_Type'] = not_completed_validated.apply(
    lambda row: 'Network Issue' if row['Had Network Issue'] else 'Engagement Issue',
    axis=1
)
```

### Step 3: Feature Selection
```python
# For Completed questions (NO network params)
completed = df_filtered[df_filtered['Attempt Status'] == 'Completed']
X_completed = completed[['Response Time (sec)', 'Is_Correct_Binary']]

# For Not Completed (USE network params)
not_completed = df_filtered[df_filtered['Attempt Status'] == 'Not Completed']
X_not_completed = not_completed[[
    'Response Time (sec)', 'RTT (ms)', 'Jitter (ms)', 
    'Stability (%)', 'Network_Quality_Encoded'
]]

# For Initial questions (baseline)
initial = df_filtered[df_filtered['Quiz#'] == 0]
X_initial = initial[[
    'Response Time (sec)', 'RTT (ms)', 'Jitter (ms)', 'Stability (%)'
]]
```

---

## Statistics

### Merge_Enhanced.csv
| Metric | Value |
|--------|-------|
| Total Records | 5,522 |
| Students | 141 |
| Initial Questions | 423 (3 per student) |
| Quiz Questions | 5,099 |
| Completed | 5,003 (90.6%) |
| Not Completed | 519 (9.4%) |
| Active | 3,211 (58.2%) |
| Passive | 1,606 (29.1%) |
| Moderate | 282 (5.1%) |

### Participant_Tracking.csv
| Metric | Value |
|--------|-------|
| Total Events | 786 |
| Join Events | 393 |
| Leave Events | 393 |
| Sessions | 393 (141 students × ~2.8 quizzes avg) |
| Students with Network Issues | 135 (95.7%) |
| Avg Completion Rate | ~90% |

---

## Download Instructions

1. **From GitHub Repository**:
   - Navigate to repository root
   - Download `Merge_Enhanced.csv` (2.9 MB)
   - Download `Participant_Tracking.csv` (124 KB)

2. **Upload to Google Drive**:
   ```
   /content/drive/MyDrive/FYP_Data/
   ├── Merge_Enhanced.csv
   └── Participant_Tracking.csv
   ```

3. **Verify in Colab**:
   ```python
   import os
   path = '/content/drive/MyDrive/FYP_Data/'
   print("Files:", os.listdir(path))
   # Should show: ['Merge_Enhanced.csv', 'Participant_Tracking.csv']
   ```

---

## Related Documentation

- **DATA_PROCESSING_RULES.txt**: Complete processing guidelines
- **DATASET_ENHANCEMENT_DOCS.md**: Enhancement methodology
- **notebooks/README.md**: Notebook usage guide
- **CHANGES_SUMMARY.md**: Summary of changes made

---

## Quick Data Quality Checks

```python
import pandas as pd

# Load datasets
df = pd.read_csv('Merge_Enhanced.csv')
participant = pd.read_csv('Participant_Tracking.csv')

# Check 1: No missing values
assert df.isnull().sum().sum() == 0, "Found missing values!"

# Check 2: All students in df are in participant tracking
students_df = set(df['Admission No'].unique())
students_part = set(participant['Admission No'].unique())
assert students_df == students_part, "Student mismatch!"

# Check 3: Initial questions exist
assert len(df[df['Quiz#'] == 0]) == 423, "Initial questions missing!"

# Check 4: Completion status exists
assert 'Not Completed' in df['Attempt Status'].values, "No not completed!"

print("✅ All quality checks passed!")
```

---

**Ready to use!** These datasets are preprocessed and ready for model training.
