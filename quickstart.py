"""
Quick Start Guide - Student Engagement Analytics System

This script demonstrates the basic usage of all implemented models.
"""

import sys
sys.path.append('src')

print("="*80)
print(" "*15 + "STUDENT ENGAGEMENT ANALYTICS SYSTEM")
print(" "*20 + "Quick Start Guide")
print("="*80)

# ============================================================================
# 1. DATA PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("1. DATA PREPROCESSING")
print("="*80)

from src.utils.data_preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor('Merge.csv')

# Load and process data
df = preprocessor.load_data()
df = preprocessor.clean_data()
df = preprocessor.engineer_features()

# Get data summary
print("\nKey Statistics:")
print(f"  Total Records: {len(df)}")
print(f"  Total Students: {df['Student Name'].nunique()}")
print(f"  Average Accuracy: {df['Is_Correct_Binary'].mean():.2%}")
print(f"  Average Response Time: {df['Response Time (sec)'].mean():.1f}s")

# ============================================================================
# 2. PREDICTIVE LEARNER CLUSTERING
# ============================================================================
print("\n" + "="*80)
print("2. PREDICTIVE LEARNER CLUSTERING")
print("="*80)

from src.models.clustering_model import PredictiveClusteringModel

# Aggregate student features
student_features = preprocessor.aggregate_student_features()
feature_cols = ['Response Time (sec)', 'RTT (ms)', 'Jitter (ms)', 
               'Stability (%)', 'Is_Correct_Binary', 'Engagement_Encoded']
X = student_features[feature_cols]

# Train model
model = PredictiveClusteringModel(n_clusters=3)
labels = model.fit_predict(X)
mapped_labels = model.map_clusters_to_engagement(X.values, labels)

# Evaluate
metrics = model.evaluate(X, mapped_labels)

# Add cluster info to student features
student_features['cluster_id'] = mapped_labels
student_features['cluster_name'] = student_features['cluster_id'].map(
    {0: 'Passive', 1: 'Moderate', 2: 'Active'}
)

print("\nCluster Distribution:")
for cluster_name, count in student_features['cluster_name'].value_counts().items():
    percentage = (count / len(student_features)) * 100
    print(f"  {cluster_name:10s}: {count:3d} students ({percentage:5.1f}%)")

# ============================================================================
# 3. TARGETED QUESTION TRIGGERING
# ============================================================================
print("\n" + "="*80)
print("3. TARGETED QUESTION TRIGGERING")
print("="*80)

from src.models.question_targeting import QuestionTriggeringModule

# Initialize module
qt_module = QuestionTriggeringModule()
qt_module.create_sample_question_bank()

# Example: Get questions for different clusters
print("\nExample Questions by Cluster:")
for cluster_id in [0, 1, 2]:
    question = qt_module.get_cluster_based_question(cluster_id)
    if question:
        print(f"\n{qt_module.cluster_labels[cluster_id]} Cluster:")
        print(f"  Question: {question['question_text'][:70]}...")
        print(f"  Difficulty: {question['difficulty']}")

# Trigger questions for a session
print("\nTriggering questions for session...")
student_clusters = dict(zip(
    student_features['Student Name'].head(5),
    student_features['cluster_id'].head(5)
))
student_accuracy = dict(zip(
    student_features['Student Name'].head(5),
    student_features['Is_Correct_Binary'].head(5)
))

triggered = qt_module.trigger_questions_for_session(
    student_clusters=student_clusters,
    student_accuracy=student_accuracy,
    num_questions=1
)

print(f"Triggered {len(triggered)} question sets")

# ============================================================================
# 4. PERSONALIZED FEEDBACK GENERATION
# ============================================================================
print("\n" + "="*80)
print("4. PERSONALIZED FEEDBACK GENERATION")
print("="*80)

from src.models.feedback_generation import FeedbackGenerator

# Initialize generator
feedback_gen = FeedbackGenerator()

# Generate feedback for sample students
print("\nExample Feedback:")
sample_students = student_features.head(3)

for idx, row in sample_students.iterrows():
    cluster_name = row['cluster_name']
    accuracy = row['Is_Correct_Binary']
    
    feedback = feedback_gen.generate_feedback(
        cluster_id=row['cluster_id'],
        accuracy=accuracy,
        network_quality='Poor'  # Simulated
    )
    
    print(f"\n{row['Student Name']}:")
    print(f"  Cluster: {cluster_name}")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Feedback: {feedback}")

# ============================================================================
# 5. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
✅ Data Preprocessing: Successfully processed {len(df)} records
✅ Clustering Model: Trained with Silhouette Score = {metrics['Silhouette Score']:.3f}
✅ Question Targeting: Created question bank with {len(qt_module.questions)} questions
✅ Feedback Generation: Generated personalized feedback

Next Steps:
1. Run full pipeline: python src/main_pipeline.py
2. Generate visualizations: python generate_visualizations.py
3. Check output files in: data/processed/
4. Review PROJECT_README.md for detailed documentation
""")

print("="*80)
print(" "*20 + "QUICK START COMPLETE")
print("="*80)
