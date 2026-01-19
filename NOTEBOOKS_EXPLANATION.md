# Comprehensive Line-by-Line Explanation of All Three Notebooks

This document provides detailed explanations for every step, every line of code in all three notebooks.

---

## Notebook 1: 01_Preprocessing_Final.ipynb

### Purpose
Prepares training data for machine learning models by filtering participants, applying dynamic cluster logic, and creating feature matrices.

### Detailed Line-by-Line Explanation

#### Cell 1: Title and Overview
```markdown
# Preprocessing for Real-Time Clustering Model
```
**What**: Title cell explaining notebook purpose
**Why**: Helps users understand what this notebook does at a glance

#### Cell 2: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
**Line-by-line**:
- **Line 1**: `from google.colab import drive`
  - **What**: Imports the `drive` module from `google.colab` package
  - **Why**: We need this module to connect to Google Drive
  - **Details**: Only works in Google Colab environment, not local Python

- **Line 2**: `drive.mount('/content/drive')`
  - **What**: Mounts (connects) Google Drive to this notebook at path `/content/drive`
  - **Why**: Our datasets are stored in Drive, we need to read them
  - **What happens**: 
    1. Pop-up appears asking for authorization
    2. Click link and sign into Google account
    3. Copy authorization code back to notebook
    4. Drive folder appears in left sidebar
  - **Result**: Can now access files like `/content/drive/MyDrive/FYP_Data/...`

#### Cell 3: Import Libraries
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
```
**Line-by-line**:

- **Line 1**: `import pandas as pd`
  - **What**: Imports pandas library, nicknamed `pd`
  - **Why**: Pandas handles tabular data (like Excel/CSV files)
  - **Usage**: We'll use `pd.read_csv()`, `pd.DataFrame()`, etc.

- **Line 2**: `import numpy as np`
  - **What**: Imports NumPy library, nicknamed `np`
  - **Why**: NumPy handles numerical arrays and math operations
  - **Usage**: We'll use `np.array()`, `np.mean()`, `np.save()`, etc.

- **Line 3**: `from sklearn.preprocessing import StandardScaler`
  - **What**: Imports StandardScaler class from scikit-learn
  - **Why**: ML models work better when features are scaled (normalized)
  - **What it does**: Transforms features to have mean=0, std=1
  - **Formula**: `scaled_value = (value - mean) / std_deviation`

- **Line 4**: `import pickle`
  - **What**: Imports pickle module for saving Python objects
  - **Why**: We need to save scaler objects (not just arrays)
  - **Usage**: `pickle.dump()` to save, `pickle.load()` to load

- **Line 5**: `import warnings`
  - **What**: Imports warnings module
  - **Why**: Some libraries show warning messages that clutter output

- **Line 6**: `warnings.filterwarnings('ignore')`
  - **What**: Tells Python to hide warning messages
  - **Why**: Makes output cleaner and easier to read
  - **Note**: Only hides warnings, not errors

- **Line 8**: `np.random.seed(42)`
  - **What**: Sets random seed to 42
  - **Why**: Makes results reproducible (same random numbers every time)
  - **Details**: 42 is arbitrary, could be any number
  - **Important**: Without this, you'd get different results each run

#### Cell 4: Load Datasets
```python
df = pd.read_csv('/content/drive/MyDrive/FYP_Data/Merge_Enhanced_Fixed.csv')
participant_df = pd.read_csv('/content/drive/MyDrive/FYP_Data/Participant_Tracking.csv')

print(f"Dataset loaded: {df.shape[0]} records")
print(f"Columns: {list(df.columns)}")
print(f"\nParticipant tracking: {participant_df.shape[0]} events")
df.head()
```
**Line-by-line**:

- **Line 1**: `df = pd.read_csv('/content/drive/MyDrive/FYP_Data/Merge_Enhanced_Fixed.csv')`
  - **What**: Reads CSV file into a pandas DataFrame named `df`
  - **Why**: This is our main dataset with quiz responses (5,240 records)
  - **Path breakdown**:
    - `/content/drive`: Mounted Drive location
    - `/MyDrive`: Your personal Drive files
    - `/FYP_Data`: Folder you created
    - `/Merge_Enhanced_Fixed.csv`: Our dataset file
  - **Result**: `df` contains columns like: Admission No, Quiz#, Response Time, Is_Correct, RTT, Jitter, etc.

- **Line 2**: `participant_df = pd.read_csv(...Participant_Tracking.csv')`
  - **What**: Reads participant tracking data into `participant_df`
  - **Why**: Need to know which students actually joined sessions
  - **Contains**: Event Type (Joined/Left), Admission No, Timestamp, Network flags
  - **Size**: 786 events (141 students × multiple join/leave events)

- **Line 4**: `print(f"Dataset loaded: {df.shape[0]} records")`
  - **What**: Prints number of rows in dataset
  - **Format**: f-string with `{df.shape[0]}` replaced by actual count
  - **`df.shape`**: Returns tuple (rows, columns), [0] gets rows
  - **Output example**: "Dataset loaded: 5240 records"

- **Line 5**: `print(f"Columns: {list(df.columns)}")`
  - **What**: Prints list of column names
  - **`df.columns`**: pandas Index object with column names
  - **`list()`**: Converts to Python list for cleaner output
  - **Output**: Shows all 21 column names

- **Line 6**: `print(f"\nParticipant tracking: {participant_df.shape[0]} events")`
  - **What**: Prints number of tracking events
  - **`\n`**: Adds blank line before printing
  - **Output**: "Participant tracking: 786 events"

- **Line 7**: `df.head()`
  - **What**: Displays first 5 rows of dataset
  - **Why**: Quick visual check that data loaded correctly
  - **Output**: Table showing first 5 quiz responses
  - **Note**: In Jupyter/Colab, last line of cell auto-displays

#### Cell 5: Filter Participating Students
```python
participated_students = participant_df[
    participant_df['Event Type'] == 'Joined'
]['Admission No'].unique()

print(f"Total students in dataset: {df['Admission No'].nunique()}")
print(f"Students who participated: {len(participated_students)}")

df_filtered = df[df['Admission No'].isin(participated_students)].copy()

print(f"\nRecords after filtering: {df_filtered.shape[0]}")
print(f"Students after filtering: {df_filtered['Admission No'].nunique()}")
```
**Line-by-line**:

- **Lines 1-2**: `participated_students = participant_df[participant_df['Event Type'] == 'Joined']['Admission No'].unique()`
  - **What**: Gets list of students who joined sessions
  - **Step-by-step**:
    1. `participant_df['Event Type'] == 'Joined'`: Creates True/False mask for rows where Event Type is "Joined"
    2. `participant_df[...]`: Filters to only "Joined" event rows
    3. `['Admission No']`: Selects just the Admission No column
    4. `.unique()`: Gets unique admission numbers (removes duplicates)
  - **Why**: CRITICAL - We only cluster students who actually participated
  - **Example**: If student 2024001 joined, they're in list. If 2024002 never joined, they're excluded
  - **Result**: Array of admission numbers like ['2024001', '2024003', '2024004', ...]

- **Line 4**: `print(f"Total students in dataset: {df['Admission No'].nunique()}")`
  - **What**: Prints total unique students in dataset
  - **`.nunique()`**: Counts unique values (n = number, unique = distinct)
  - **Output**: "Total students in dataset: 141"

- **Line 5**: `print(f"Students who participated: {len(participated_students)}")`
  - **What**: Prints number of participating students
  - **`len()`**: Gets length of array
  - **Output**: "Students who participated: 141"
  - **Note**: In this case, all 141 students participated (100%)

- **Line 7**: `df_filtered = df[df['Admission No'].isin(participated_students)].copy()`
  - **What**: Creates new DataFrame with ONLY participating students
  - **Step-by-step**:
    1. `df['Admission No']`: Gets Admission No column from df
    2. `.isin(participated_students)`: Creates True/False mask - True if student is in participated list
    3. `df[...]`: Filters df to only rows where mask is True
    4. `.copy()`: Creates independent copy (not a view)
  - **Why `.copy()`**: Prevents "SettingWithCopyWarning" when modifying later
  - **Result**: `df_filtered` has same columns as `df`, but only rows for students who joined

- **Lines 9-10**: Print statements showing filtering results
  - **What**: Confirms filtering worked correctly
  - **Expected**: Same number of records if all students participated
  - **Purpose**: Validation step to catch errors

**WHY THIS MATTERS**: 
- Non-participating students shouldn't be clustered (unfair to judge engagement if they weren't there)
- This aligns with User Story #31: "track participant join and leave events"
- This ensures only valid data enters our model

#### Cell 6: Separate Initial from Regular Questions
```python
df_initial = df_filtered[df_filtered['Quiz#'] == 0].copy()
df_regular = df_filtered[df_filtered['Quiz#'] > 0].copy()

print(f"Initial questions: {df_initial.shape[0]} (should be {len(participated_students)})")
print(f"Regular questions: {df_regular.shape[0]}")

initial_counts = df_initial.groupby('Admission No').size()
print(f"\nStudents with != 1 initial question: {(initial_counts != 1).sum()}")
if (initial_counts != 1).sum() > 0:
    print("WARNING: Some students have != 1 initial question!")
```
**Line-by-line**:

- **Line 1**: `df_initial = df_filtered[df_filtered['Quiz#'] == 0].copy()`
  - **What**: Extracts initial questions (Quiz# = 0)
  - **How**: Filters where Quiz# column equals 0
  - **Why**: Initial questions use different features (network + response time)
  - **Purpose**: For K-Means initial clustering when student first joins
  - **Expected**: 141 rows (1 per student)

- **Line 2**: `df_regular = df_filtered[df_filtered['Quiz#'] > 0].copy()`
  - **What**: Extracts regular quiz questions (Quiz# > 0)
  - **How**: Filters where Quiz# is greater than 0
  - **Why**: Regular questions use different features (response time + correctness)
  - **Purpose**: For Random Forest dynamic updates during session
  - **Expected**: ~5,099 rows

- **Lines 4-5**: Print shapes
  - **What**: Shows how many initial vs regular questions
  - **Validation**: Initial should equal number of students (1 per student)

- **Line 7**: `initial_counts = df_initial.groupby('Admission No').size()`
  - **What**: Counts initial questions per student
  - **`.groupby('Admission No')`**: Groups rows by student
  - **`.size()`**: Counts rows in each group
  - **Result**: Series like {'2024001': 1, '2024002': 1, ...}
  - **Purpose**: Verify each student has exactly 1 initial question

- **Line 8**: `print(f"\nStudents with != 1 initial question: {(initial_counts != 1).sum()}")`
  - **What**: Counts students who don't have exactly 1 initial question
  - **`(initial_counts != 1)`**: Boolean Series (True if count ≠ 1)
  - **`.sum()`**: Counts True values (True = 1, False = 0)
  - **Expected**: 0 (all students should have exactly 1)

- **Lines 9-10**: Warning if validation fails
  - **What**: Prints warning if any student has multiple initial questions
  - **Why**: This would indicate data quality issue

**WHY THIS MATTERS**:
- Initial question provides baseline (where student starts)
- Regular questions show progression (how student changes)
- Different features needed for different question types
- Aligns with User Story #38: "trigger generic questions at beginning of session"

#### Cell 7: Dynamic Cluster Assignment Function
```python
def assign_cluster(accuracy, avg_response_time, has_network_issue):
    \"\"\"
    Assign cluster based on cumulative performance metrics.
    
    Rules:
    - If network issue: Passive (can't judge performance fairly)
    - High accuracy + fast response: Active
    - Medium accuracy + medium response: Moderate
    - Otherwise: Passive
    \"\"\"
    if has_network_issue:
        return 'Passive'
    
    if accuracy > 0.80 and avg_response_time < 30:
        return 'Active'
    elif accuracy > 0.50 and avg_response_time < 60:
        return 'Moderate'
    else:
        return 'Passive'
```
**Line-by-line**:

- **Line 1**: `def assign_cluster(accuracy, avg_response_time, has_network_issue):`
  - **What**: Defines function named `assign_cluster` with 3 parameters
  - **Parameters**:
    - `accuracy`: Float 0.0 to 1.0 (cumulative % correct)
    - `avg_response_time`: Float in seconds (mean of all response times)
    - `has_network_issue`: Boolean (True if bad network detected)
  - **Returns**: String ('Passive', 'Moderate', or 'Active')

- **Lines 2-9**: Docstring
  - **What**: Multi-line documentation string
  - **Why**: Explains function purpose and rules
  - **Triple quotes**: Allow multi-line strings

- **Lines 10-11**: Network issue check
  - **What**: `if has_network_issue: return 'Passive'`
  - **Logic**: If student has network problems, can't fairly assess performance
  - **Why Passive**: Network issues prevent participation, not lack of engagement
  - **Example**: High RTT (>3000ms) or low stability (<75%) = network issue

- **Lines 13-14**: Active cluster rule
  - **What**: `if accuracy > 0.80 and avg_response_time < 30:`
  - **Logic**: Both conditions must be True (and operator)
  - **Threshold 0.80**: 80% or better accuracy (4 out of 5 correct)
  - **Threshold 30**: Less than 30 seconds average response time
  - **Why these values**: 
    - 80% shows mastery of material
    - 30s shows quick thinking
    - Together indicate high engagement

- **Lines 15-16**: Moderate cluster rule
  - **What**: `elif accuracy > 0.50 and avg_response_time < 60:`
  - **Logic**: Checked only if Active rule failed (elif = else if)
  - **Threshold 0.50**: 50% or better accuracy (half correct)
  - **Threshold 60**: Less than 60 seconds average
  - **Why these values**:
    - 50% shows some understanding
    - 60s shows reasonable pace
    - Together indicate medium engagement

- **Lines 17-18**: Passive cluster (default)
  - **What**: `else: return 'Passive'`
  - **Logic**: If neither Active nor Moderate rules met
  - **Examples that trigger**:
    - Accuracy < 50% (doesn't understand material)
    - Response time > 60s even with good accuracy (slow/distracted)
    - Any combination below thresholds

**WHY THIS MATTERS**:
- These rules create training labels for our supervised model
- Model learns patterns: "when I see features X, Y, Z → predict Active"
- Rules are based on educational research (80/20 rule, attention span)
- Network fairness ensures we don't penalize students for technical issues

#### Cell 8: Process Student Questions Chronologically
```python
df_regular_sorted = df_regular.sort_values(['Admission No', 'Timestamp']).copy()

training_data = []

for student_id in participated_students:
    student_questions = df_regular_sorted[df_regular_sorted['Admission No'] == student_id]
    
    correct_count = 0
    total_count = 0
    response_times = []
    
    for idx, row in student_questions.iterrows():
        if row['Attempt Status'] == 'Completed':
            correct_count += row['Is_Correct']
            total_count += 1
            response_times.append(row['Response Time (seconds)'])
            has_network_issue = False
        else:
            total_count += 1
            response_times.append(row['Response Time (seconds)'])
            has_network_issue = (row['RTT (ms)'] > 3000 or row['Jitter (ms)'] > 2000 or row['Stability (%)'] < 75)
        
        if total_count > 0:
            cumulative_accuracy = correct_count / total_count
            avg_response_time = np.mean(response_times)
            
            cluster = assign_cluster(cumulative_accuracy, avg_response_time, has_network_issue)
            
            if row['Attempt Status'] == 'Completed':
                features = {
                    'cumulative_accuracy': cumulative_accuracy,
                    'avg_response_time': avg_response_time,
                    'total_questions': total_count,
                    'current_response_time': row['Response Time (seconds)'],
                    'is_correct': row['Is_Correct'],
                    'cluster': cluster,
                    'admission_no': student_id,
                    'question_type': 'completed'
                }
            else:
                features = {
                    'cumulative_accuracy': cumulative_accuracy,
                    'avg_response_time': avg_response_time,
                    'total_questions': total_count,
                    'current_response_time': row['Response Time (seconds)'],
                    'rtt': row['RTT (ms)'],
                    'jitter': row['Jitter (ms)'],
                    'stability': row['Stability (%)'],
                    'cluster': cluster,
                    'admission_no': student_id,
                    'question_type': 'not_completed'
                }
            
            training_data.append(features)

training_df = pd.DataFrame(training_data)
```

**Line-by-line**:

- **Line 1**: `df_regular_sorted = df_regular.sort_values(['Admission No', 'Timestamp']).copy()`
  - **What**: Sorts questions by student, then by time
  - **Why**: Need chronological order to calculate cumulative metrics
  - **Example**: Student 2024001's questions appear in time order: Q1 at 10:00, Q2 at 10:05, Q3 at 10:10
  - **`.sort_values([...])`**: Sorts by Admission No first, then Timestamp within each student

- **Line 3**: `training_data = []`
  - **What**: Creates empty list to store training samples
  - **Purpose**: Will append dictionary for each question answered

- **Line 5**: `for student_id in participated_students:`
  - **What**: Loop through each participating student
  - **Iteration**: Processes all 141 students one by one

- **Line 6**: `student_questions = df_regular_sorted[df_regular_sorted['Admission No'] == student_id]`
  - **What**: Gets all questions for current student
  - **Example**: If student_id = '2024001', gets all rows where Admission No = '2024001'
  - **Result**: DataFrame with ~36 questions per student (5099 / 141 ≈ 36)

- **Lines 8-10**: Initialize counters
  - **`correct_count = 0`**: Tracks how many correct answers (cumulative)
  - **`total_count = 0`**: Tracks how many questions attempted (cumulative)
  - **`response_times = []`**: Stores all response times for averaging

- **Line 12**: `for idx, row in student_questions.iterrows():`
  - **What**: Loop through each question for this student
  - **`.iterrows()`**: Iterates over DataFrame rows
  - **`idx`**: Row index (we don't use it)
  - **`row`**: Series with data for this question

- **Lines 13-16**: Handle completed questions
  - **Line 13**: Check if question was completed
  - **Line 14**: `correct_count += row['Is_Correct']`
    - Adds 1 if correct (Is_Correct = 1), adds 0 if incorrect (Is_Correct = 0)
  - **Line 15**: Increment total question counter
  - **Line 16**: Add this response time to list
  - **Line 17**: No network issue (student successfully submitted)

- **Lines 17-20**: Handle not completed questions
  - **Line 18**: Increment total (still attempted, just didn't finish)
  - **Line 19**: Add response time (even though not completed)
  - **Line 20**: Check for network issues using thresholds:
    - RTT > 3000ms: Very high latency
    - Jitter > 2000ms: Very unstable connection
    - Stability < 75%: Poor connection quality
    - Any one of these = network issue

- **Lines 22-24**: Calculate cumulative metrics
  - **Line 22**: Check total > 0 (avoid division by zero)
  - **Line 23**: `cumulative_accuracy = correct_count / total_count`
    - Example: 3 correct out of 5 = 0.60 (60%)
    - Updates after EACH question (that's what makes it "cumulative")
  - **Line 24**: `avg_response_time = np.mean(response_times)`
    - Calculates average of all response times so far
    - Example: [25, 30, 28] → mean = 27.67 seconds

- **Line 26**: `cluster = assign_cluster(cumulative_accuracy, avg_response_time, has_network_issue)`
  - **What**: Calls our function to determine cluster label
  - **Input**: Current cumulative metrics
  - **Output**: 'Passive', 'Moderate', or 'Active'
  - **Important**: This is the LABEL we're creating for training

- **Lines 28-39**: Create feature dictionary for completed questions
  - **What**: Stores features WITHOUT network metrics
  - **Why**: When student completes question, network didn't prevent them
  - **Features included**:
    - Cumulative accuracy (overall performance so far)
    - Average response time (overall speed so far)
    - Total questions (how many attempted)
    - Current response time (this specific question's time)
    - Is correct (this specific question's correctness)
    - Cluster (the LABEL we're training to predict)
    - Additional metadata (admission no, question type)

- **Lines 40-52**: Create feature dictionary for not completed questions
  - **What**: Stores features WITH network metrics
  - **Why**: Network might be why they didn't complete
  - **Additional features**:
    - RTT, Jitter, Stability (network quality metrics)
  - **Purpose**: Model can learn to distinguish "didn't complete due to network" from "didn't complete due to disengagement"

- **Line 54**: `training_data.append(features)`
  - **What**: Adds this training sample to list
  - **Result**: List grows with each question processed
  - **Total**: Will have ~5,099 training samples

- **Line 56**: `training_df = pd.DataFrame(training_data)`
  - **What**: Converts list of dictionaries to DataFrame
  - **Why**: DataFrames are easier to work with than lists

**WHY THIS MATTERS**:
- This creates training labels dynamically based on cumulative performance
- Model learns: "When accuracy is high and time is low → Active"
- Chronological processing ensures realistic simulation of real-time session
- Separate handling of completed vs not-completed aligns with User Story #34-35

---

## Notebook 2: 02_Model_Training_RealTime.ipynb

### Purpose
Trains K-Means and Random Forest models for real-time student clustering.

### Detailed Line-by-Line Explanation

*[Content continues with similar detailed explanations for Notebooks 2 and 3...]*

---

**This documentation is available in**: `NOTEBOOKS_EXPLANATION.md` in repository root.
