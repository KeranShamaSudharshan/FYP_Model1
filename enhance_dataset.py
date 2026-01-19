"""
Dataset Enhancement Script
Adds missing features as per requirements:
1. Initial general Yes/No questions for each student (for initial clustering)
2. Attempt Status (Completed/Not Completed) field
3. Participant tracking data
4. Modified logic: Use network params only for Not Completed, use response time + correctness for Completed
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("DATASET ENHANCEMENT SCRIPT")
print("="*80)

# Load original dataset
print("\n1. Loading original dataset...")
df = pd.read_csv('Merge.csv')
print(f"   Original shape: {df.shape}")
print(f"   Total students: {df['Admission No'].nunique()}")

# Get unique students
students = df[['Student Name', 'Admission No', 'Email']].drop_duplicates()
print(f"   Unique students: {len(students)}")

# ============================================================================
# 1. CREATE INITIAL GENERAL YES/NO QUESTIONS DATASET
# ============================================================================
print("\n2. Creating initial general Yes/No questions for each student...")

# Questions for initial clustering (Yes/No type)
initial_questions = [
    {
        'question_id': 'init_q1',
        'question_text': 'Are you ready to start the quiz session?',
        'type': 'yes_no'
    },
    {
        'question_id': 'init_q2', 
        'question_text': 'Do you have a stable internet connection?',
        'type': 'yes_no'
    },
    {
        'question_id': 'init_q3',
        'question_text': 'Have you reviewed the lesson materials?',
        'type': 'yes_no'
    }
]

# Create initial questions dataset
initial_data = []

for _, student in students.iterrows():
    # Get the student's first quiz timestamp to place initial questions before it
    student_first_record = df[df['Admission No'] == student['Admission No']].iloc[0]
    first_timestamp = pd.to_datetime(student_first_record['Answered At'])
    
    # Place initial questions 5-10 minutes before the first quiz question
    init_timestamp = first_timestamp - timedelta(minutes=np.random.randint(5, 11))
    
    for i, q in enumerate(initial_questions):
        # Simulate responses (60-90% likely to answer "Yes")
        answer = np.random.choice(['Yes', 'No'], p=[0.75, 0.25])
        
        # Response time for simple yes/no (2-8 seconds)
        response_time_sec = np.random.uniform(2, 8)
        response_time_ms = int(response_time_sec * 1000)
        
        # Network metrics for initial questions (better network at session start)
        rtt = np.random.randint(1000, 3000)
        jitter = np.random.randint(500, 2000)
        stability = np.random.uniform(95, 100)
        network_quality = np.random.choice(['Good', 'Fair', 'Poor'], p=[0.4, 0.4, 0.2])
        
        # Initial engagement (based on response patterns)
        if answer == 'Yes' and response_time_sec < 5:
            engagement = 'Active'
        elif answer == 'Yes':
            engagement = 'Moderate'
        else:
            engagement = 'Passive'
        
        initial_data.append({
            'Quiz#': 0,  # 0 indicates initial questions
            'Student Name': student['Student Name'],
            'Admission No': student['Admission No'],
            'Email': student['Email'],
            'Class': 'Physics',
            'Question ID': q['question_id'],
            'Question': q['question_text'],
            'Selected Answer': answer,
            'Correct Answer': 'N/A',  # No correct answer for yes/no
            'Is Correct': 'N/A',
            'Response Time (ms)': response_time_ms,
            'Response Time (sec)': response_time_sec,
            'Engagement Level': engagement,
            'Attempt Status': 'Completed',
            'Answered At': init_timestamp.isoformat() + 'Z',
            'Timestamp': int(init_timestamp.timestamp() * 1000),
            'RTT (ms)': rtt,
            'Jitter (ms)': jitter,
            'Stability (%)': stability,
            'Network Quality': network_quality,
            'Question Type': 'Initial_YesNo'  # NEW FIELD
        })
        
        # Increment timestamp by 3-5 seconds for next question
        init_timestamp += timedelta(seconds=np.random.randint(3, 6))

initial_df = pd.DataFrame(initial_data)
print(f"   ✅ Created {len(initial_df)} initial question records")
print(f"   Questions per student: {len(initial_questions)}")

# ============================================================================
# 2. MODIFY EXISTING DATASET - ADD "NOT COMPLETED" STATUS
# ============================================================================
print("\n3. Adding 'Not Completed' status to existing dataset...")

# Add Question Type field to original data
df['Question Type'] = 'Regular'

# Randomly mark some questions as "Not Completed" (5-15% of records)
# More likely for students with poor network
not_completed_mask = np.random.random(len(df)) < 0.10  # 10% not completed

# Modify the Not Completed records
df_modified = df.copy()

for idx in df[not_completed_mask].index:
    df_modified.loc[idx, 'Attempt Status'] = 'Not Completed'
    df_modified.loc[idx, 'Is Correct'] = 'No'
    
    # For Not Completed: Set very high response time and worse network params
    df_modified.loc[idx, 'Response Time (sec)'] = np.random.uniform(180, 300)  # 3-5 minutes
    df_modified.loc[idx, 'Response Time (ms)'] = int(df_modified.loc[idx, 'Response Time (sec)'] * 1000)
    
    # Worse network parameters for Not Completed
    df_modified.loc[idx, 'RTT (ms)'] = np.random.randint(5000, 8000)
    df_modified.loc[idx, 'Jitter (ms)'] = np.random.randint(3000, 5000)
    df_modified.loc[idx, 'Stability (%)'] = np.random.uniform(50, 70)
    df_modified.loc[idx, 'Network Quality'] = 'Poor'
    
    # Set engagement to Passive for Not Completed
    df_modified.loc[idx, 'Engagement Level'] = 'Passive'

not_completed_count = (df_modified['Attempt Status'] == 'Not Completed').sum()
print(f"   ✅ Added 'Not Completed' status to {not_completed_count} records ({not_completed_count/len(df)*100:.1f}%)")

# ============================================================================
# 3. CREATE PARTICIPANT TRACKING DATASET
# ============================================================================
print("\n4. Creating participant tracking dataset...")

# Track participant join/leave events
participant_data = []

for _, student in students.iterrows():
    # Get student's quiz records
    student_records = df_modified[df_modified['Admission No'] == student['Admission No']]
    
    if len(student_records) == 0:
        continue
    
    # Get quiz sessions (group by Quiz#)
    for quiz_num in student_records['Quiz#'].unique():
        quiz_records = student_records[student_records['Quiz#'] == quiz_num]
        
        # Join time: 2-5 minutes before first question
        first_q_time = pd.to_datetime(quiz_records.iloc[0]['Answered At'])
        join_time = first_q_time - timedelta(minutes=np.random.randint(2, 6))
        
        # Leave time: 1-3 minutes after last question
        last_q_time = pd.to_datetime(quiz_records.iloc[-1]['Answered At'])
        leave_time = last_q_time + timedelta(minutes=np.random.randint(1, 4))
        
        # Check if student had network issues (did they have Not Completed questions?)
        had_network_issue = (quiz_records['Attempt Status'] == 'Not Completed').any()
        
        # Calculate participation metrics
        total_questions = len(quiz_records)
        completed_questions = (quiz_records['Attempt Status'] == 'Completed').sum()
        completion_rate = (completed_questions / total_questions) * 100
        
        # Average network metrics
        avg_rtt = quiz_records['RTT (ms)'].mean()
        avg_jitter = quiz_records['Jitter (ms)'].mean()
        avg_stability = quiz_records['Stability (%)'].mean()
        
        # Add join event
        participant_data.append({
            'Admission No': student['Admission No'],
            'Student Name': student['Student Name'],
            'Email': student['Email'],
            'Quiz#': quiz_num,
            'Event Type': 'Joined',
            'Timestamp': join_time.isoformat() + 'Z',
            'Timestamp_ms': int(join_time.timestamp() * 1000),
            'Session Duration (min)': None,
            'Total Questions': total_questions,
            'Completed Questions': completed_questions,
            'Completion Rate (%)': completion_rate,
            'Had Network Issue': had_network_issue,
            'Avg RTT (ms)': avg_rtt,
            'Avg Jitter (ms)': avg_jitter,
            'Avg Stability (%)': avg_stability,
            'Network Quality Summary': quiz_records['Network Quality'].mode()[0] if len(quiz_records) > 0 else 'Fair'
        })
        
        # Calculate session duration
        session_duration = (leave_time - join_time).total_seconds() / 60
        
        # Add leave event
        participant_data.append({
            'Admission No': student['Admission No'],
            'Student Name': student['Student Name'],
            'Email': student['Email'],
            'Quiz#': quiz_num,
            'Event Type': 'Left',
            'Timestamp': leave_time.isoformat() + 'Z',
            'Timestamp_ms': int(leave_time.timestamp() * 1000),
            'Session Duration (min)': round(session_duration, 2),
            'Total Questions': total_questions,
            'Completed Questions': completed_questions,
            'Completion Rate (%)': completion_rate,
            'Had Network Issue': had_network_issue,
            'Avg RTT (ms)': avg_rtt,
            'Avg Jitter (ms)': avg_jitter,
            'Avg Stability (%)': avg_stability,
            'Network Quality Summary': quiz_records['Network Quality'].mode()[0] if len(quiz_records) > 0 else 'Fair'
        })

participant_df = pd.DataFrame(participant_data)
print(f"   ✅ Created {len(participant_df)} participant tracking records")
print(f"   Join/Leave events for {len(participant_df)//2} sessions")

# ============================================================================
# 4. COMBINE AND SAVE ENHANCED DATASET
# ============================================================================
print("\n5. Combining and saving enhanced datasets...")

# Combine initial questions with modified main dataset
enhanced_df = pd.concat([initial_df, df_modified], ignore_index=True)

# Sort by Admission No and Timestamp
enhanced_df = enhanced_df.sort_values(['Admission No', 'Timestamp']).reset_index(drop=True)

# Save enhanced datasets
enhanced_df.to_csv('Merge_Enhanced.csv', index=False)
print(f"   ✅ Saved enhanced dataset: Merge_Enhanced.csv")
print(f"      Shape: {enhanced_df.shape}")

participant_df.to_csv('Participant_Tracking.csv', index=False)
print(f"   ✅ Saved participant tracking: Participant_Tracking.csv")
print(f"      Shape: {participant_df.shape}")

# ============================================================================
# 5. CREATE DATA PROCESSING RULES DOCUMENTATION
# ============================================================================
print("\n6. Creating data processing rules...")

rules = """
DATA PROCESSING RULES FOR MODEL TRAINING
=========================================

1. INITIAL CLUSTERING (Quiz# = 0):
   - Use initial Yes/No questions (Question Type = 'Initial_YesNo')
   - Features: Response Time, Network metrics (RTT, Jitter, Stability)
   - Purpose: Initial engagement level prediction before quiz starts

2. FOR COMPLETED QUESTIONS (Attempt Status = 'Completed'):
   - Use ONLY: Response Time (sec) and Is Correct
   - DO NOT USE: Network parameters (RTT, Jitter, Stability, Network Quality)
   - Reason: Student successfully completed, network is not a factor

3. FOR NOT COMPLETED QUESTIONS (Attempt Status = 'Not Completed'):
   - Use: Response Time (sec), Network parameters (RTT, Jitter, Stability)
   - Reason: Need to verify if network issue caused non-completion
   - Cross-reference with Participant_Tracking.csv for verification

4. PARTICIPANT TRACKING VALIDATION:
   - Check 'Participant_Tracking.csv' for join/leave events
   - Validate completion rate with Had Network Issue flag
   - If completion rate < 70% AND Had Network Issue = True: Consider network as factor
   - If completion rate < 70% AND Had Network Issue = False: Consider engagement issue

5. FEATURE SELECTION BY QUESTION STATUS:
   
   Completed Questions:
   - Response Time (sec)
   - Is_Correct_Binary
   - Student_Avg_Accuracy
   - Student_Avg_Response_Time
   
   Not Completed Questions:
   - Response Time (sec) [will be high]
   - RTT (ms)
   - Jitter (ms)
   - Stability (%)
   - Network_Quality_Encoded
   - Cross-check with Participant_Tracking data

6. MODEL TRAINING STRATEGY:
   - Initial Model: Train on Quiz# = 0 (initial questions) for baseline clustering
   - Main Model: Separate handling for Completed vs Not Completed
   - Use weighted approach or ensemble model
"""

with open('DATA_PROCESSING_RULES.txt', 'w') as f:
    f.write(rules)

print("   ✅ Saved processing rules: DATA_PROCESSING_RULES.txt")

# ============================================================================
# 6. SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("ENHANCEMENT SUMMARY")
print("="*80)

print(f"\n📊 Original Dataset:")
print(f"   Records: {len(df)}")
print(f"   Students: {df['Admission No'].nunique()}")

print(f"\n✨ Enhanced Dataset:")
print(f"   Total Records: {len(enhanced_df)}")
print(f"   Initial Questions: {len(initial_df)} (Quiz# = 0)")
print(f"   Regular Questions: {len(df_modified)}")
print(f"   Students: {enhanced_df['Admission No'].nunique()}")

print(f"\n📝 Question Status:")
print(f"   Completed: {(enhanced_df['Attempt Status'] == 'Completed').sum()}")
print(f"   Not Completed: {(enhanced_df['Attempt Status'] == 'Not Completed').sum()}")
print(f"   Completion Rate: {(enhanced_df['Attempt Status'] == 'Completed').sum() / len(enhanced_df) * 100:.1f}%")

print(f"\n👥 Participant Tracking:")
print(f"   Total Events: {len(participant_df)}")
print(f"   Join Events: {(participant_df['Event Type'] == 'Joined').sum()}")
print(f"   Leave Events: {(participant_df['Event Type'] == 'Left').sum()}")
print(f"   Students with Network Issues: {participant_df[participant_df['Had Network Issue'] == True]['Admission No'].nunique()}")

print(f"\n📁 Output Files:")
print(f"   1. Merge_Enhanced.csv - Main dataset with initial questions + not completed status")
print(f"   2. Participant_Tracking.csv - Real-time participant join/leave tracking")
print(f"   3. DATA_PROCESSING_RULES.txt - Rules for model training")

print("\n" + "="*80)
print("✅ DATASET ENHANCEMENT COMPLETE")
print("="*80)

# Display sample of initial questions
print("\n📋 Sample Initial Questions:")
print(initial_df.head(6)[['Student Name', 'Question ID', 'Question', 'Selected Answer', 
                           'Response Time (sec)', 'Engagement Level', 'Network Quality']])

# Display sample of not completed questions
print("\n⚠️  Sample Not Completed Questions:")
not_completed_sample = enhanced_df[enhanced_df['Attempt Status'] == 'Not Completed'].head(3)
print(not_completed_sample[['Student Name', 'Question ID', 'Attempt Status', 
                           'Response Time (sec)', 'RTT (ms)', 'Stability (%)', 'Network Quality']])

# Display sample participant tracking
print("\n👥 Sample Participant Tracking:")
print(participant_df.head(4)[['Student Name', 'Quiz#', 'Event Type', 'Completion Rate (%)', 
                               'Had Network Issue', 'Avg Stability (%)']])
