# Data Leakage Issue and Fix

## Issue Identified

**User Report**: Random Forest achieving 99.71% accuracy on test set - suspiciously high performance indicating data leakage.

## Root Cause Analysis

### The Leakage Problem

The preprocessing code creates training samples with **cumulative metrics that include the current question** being predicted. This is a classic data leakage scenario.

**Example of Leakage**:
```python
Student answers 5 questions: Q1, Q2, Q3, Q4, Q5

For Q5 prediction:
- cumulative_accuracy = 4/5 = 80%  ❌ Includes Q5's answer!
- avg_response_time = mean([20, 30, 25, 40, 15])  ❌ Includes Q5's time!
- is_correct = 1  ❌ This IS the answer we're predicting!
```

The model sees the answer before making a prediction, making it trivially easy to achieve 99%+ accuracy.

### Why This Is Wrong

**In Real-Time Production**:
- When a student is answering Q5, we DON'T know if they'll get it right
- We DON'T know their response time yet
- We can ONLY use information from Q1-Q4 (previous questions)

**Training should mirror inference**: If production can only use previous data, training must too.

## The Fix

### Correct Approach: Use Only Previous Questions

```python
# BEFORE (WRONG - Data Leakage):
for idx, row in student_questions.iterrows():
    # Update metrics (includes current question)
    correct_count += row['Is_Correct']
    total_count += 1
    response_times.append(row['Response Time'])
    
    # Calculate cumulative (INCLUDES current!)
    cumulative_accuracy = correct_count / total_count  ❌ LEAKAGE
    avg_response_time = np.mean(response_times)  ❌ LEAKAGE
    
    # Create features
    features = {
        'cumulative_accuracy': cumulative_accuracy,  # LEAKED
        'avg_response_time': avg_response_time,      # LEAKED
        'is_correct': row['Is_Correct'],             # DIRECT LEAKAGE!
        ...
    }
    training_data.append(features)
```

```python
# AFTER (CORRECT - No Leakage):
for idx, row in student_questions.iterrows():
    # Calculate metrics using ONLY previous questions
    if total_count > 0:
        cumulative_accuracy = correct_count / total_count  ✅ From Q1-Q(n-1)
        avg_response_time = np.mean(response_times)  ✅ From Q1-Q(n-1)
    else:
        # First question: use defaults
        cumulative_accuracy = 0.0
        avg_response_time = 0.0
    
    # Assign cluster based on PREVIOUS performance
    cluster = assign_cluster(cumulative_accuracy, avg_response_time, has_network_issue)
    
    # Create features WITHOUT current question's answer
    features = {
        'prev_accuracy': cumulative_accuracy,        # ✅ Previous only
        'prev_avg_time': avg_response_time,          # ✅ Previous only
        'total_questions_so_far': total_count,       # ✅ Count before current
        'current_response_time': row['Response Time'], # ✅ OK - available at submission
        # REMOVED: 'is_correct' - this is what we're predicting!
        'cluster': cluster  # Label to predict
    }
    training_data.append(features)
    
    # NOW update metrics for next iteration
    if row['Attempt Status'] == 'Completed':
        correct_count += row['Is_Correct']
        total_count += 1
        response_times.append(row['Response Time'])
```

### Key Changes

1. **Calculate metrics BEFORE processing current question**
2. **Remove `is_correct` from features** - that's the target!
3. **Update metrics AFTER creating training sample**
4. **Rename features** to clarify they're from previous questions:
   - `cumulative_accuracy` → `prev_accuracy`
   - `avg_response_time` → `prev_avg_time`
   - `total_questions` → `total_questions_so_far`

## Updated Feature Set

### For Completed Questions:
```python
X = [
    'prev_accuracy',           # Accuracy on Q1 to Q(n-1)
    'prev_avg_time',          # Avg time on Q1 to Q(n-1)
    'total_questions_so_far', # Number of previous questions
    'current_response_time'   # Time for current question (available at submission)
]
y = 'cluster'  # Passive/Moderate/Active
```

### For Not Completed Questions:
```python
X = [
    'prev_accuracy',           # Accuracy on previous questions
    'prev_avg_time',          # Avg time on previous questions
    'total_questions_so_far', # Number of previous questions
    'current_response_time',  # Time spent before not completing
    'rtt',                    # Network quality
    'jitter',
    'stability'
]
y = 'cluster'  # Usually Passive due to network/engagement issue
```

## Expected Results After Fix

**Before Fix (With Leakage)**:
- Accuracy: 99.71% ❌ Unrealistic
- Precision: 99.71% ❌ Too perfect
- F1-Score: 99.71% ❌ Suspiciously high

**After Fix (No Leakage)**:
- Accuracy: 75-85% ✅ Realistic
- Precision: 70-80% ✅ Normal for multi-class
- F1-Score: 70-80% ✅ Appropriate
- Some misclassifications ✅ Expected

## Why Lower Accuracy Is Actually Better

**With Leakage (99.71%)**:
- Model memorizes answers, not patterns
- Fails in production (real-time prediction impossible)
- Useless for actual deployment

**Without Leakage (75-85%)**:
- Model learns genuine engagement patterns
- Works in production (predicts from available data)
- Useful for instructors to identify at-risk students
- Room for improvement with better features/algorithms

## Validation Checklist

✅ **Temporal Ordering**: Features only use past data  
✅ **Target Separation**: Label not included in features  
✅ **Production Mirror**: Training matches inference conditions  
✅ **Reasonable Performance**: Accuracy reflects task difficulty  

## Implementation

The fix will be implemented in:
1. `01_Preprocessing_Final.ipynb` - Update preprocessing logic
2. `02_Model_Training_RealTime.ipynb` - Update feature names
3. `03_RealTime_Inference_Demo.ipynb` - Update inference code
4. `NOTEBOOKS_EXPLANATION.md` - Update documentation

## Testing the Fix

Run the notebooks after applying fix and verify:
1. Accuracy drops to 75-85% range ✅
2. Confusion matrix shows reasonable misclassifications ✅
3. Model still generalizes (cross-validation similar to test) ✅
4. Feature importance makes sense (prev_accuracy should be top) ✅

---

**Status**: Issue identified, fix documented, implementation pending.
