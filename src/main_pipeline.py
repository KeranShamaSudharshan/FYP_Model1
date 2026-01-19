"""
Main Pipeline - Student Engagement Analytics System
Integrates all components: Data Processing, Clustering, Question Targeting, and Feedback Generation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from utils.data_preprocessing import DataPreprocessor
from models.clustering_model import PredictiveClusteringModel
from models.question_targeting import QuestionTriggeringModule
from models.feedback_generation import FeedbackGenerator
import warnings
warnings.filterwarnings('ignore')


class StudentEngagementPipeline:
    """End-to-end pipeline for student engagement analytics"""
    
    def __init__(self, data_path='../Merge.csv'):
        """
        Initialize the pipeline
        
        Args:
            data_path: Path to the dataset
        """
        self.data_path = data_path
        self.preprocessor = DataPreprocessor(data_path)
        self.clustering_model = None
        self.question_module = None
        self.feedback_generator = None
        
        print("="*80)
        print(" "*20 + "STUDENT ENGAGEMENT ANALYTICS SYSTEM")
        print("="*80)
        print("Components:")
        print("  1. Data Preprocessing")
        print("  2. Predictive Learner Clustering (K-Means)")
        print("  3. Targeted Question Triggering")
        print("  4. Personalized Feedback Generation")
        print("="*80)
    
    def run_data_preprocessing(self):
        """Step 1: Load and preprocess data"""
        print("\n" + "="*80)
        print("STEP 1: DATA PREPROCESSING")
        print("="*80)
        
        # Load data
        self.preprocessor.load_data()
        
        # Clean data
        self.preprocessor.clean_data()
        
        # Engineer features
        self.preprocessor.engineer_features()
        
        # Get summary
        self.preprocessor.get_data_summary()
        
        return self.preprocessor.df
    
    def run_clustering(self, n_clusters=3):
        """Step 2: Train and evaluate clustering model"""
        print("\n" + "="*80)
        print("STEP 2: PREDICTIVE LEARNER CLUSTERING")
        print("="*80)
        
        # Aggregate student features
        student_features = self.preprocessor.aggregate_student_features()
        
        feature_cols = ['Response Time (sec)', 'RTT (ms)', 'Jitter (ms)', 
                       'Stability (%)', 'Is_Correct_Binary', 'Engagement_Encoded']
        X = student_features[feature_cols]
        
        # Initialize and train clustering model
        self.clustering_model = PredictiveClusteringModel(n_clusters=n_clusters)
        
        # Perform elbow method analysis
        print("\nPerforming Elbow Method Analysis...")
        self.clustering_model.elbow_method(X, max_clusters=10)
        
        # Train model
        labels = self.clustering_model.fit_predict(X)
        
        # Map to engagement levels
        mapped_labels = self.clustering_model.map_clusters_to_engagement(X.values, labels)
        student_features['cluster_id'] = mapped_labels
        student_features['cluster_name'] = student_features['cluster_id'].map(
            {0: 'Passive', 1: 'Moderate', 2: 'Active'}
        )
        
        # Evaluate model
        metrics = self.clustering_model.evaluate(X, mapped_labels)
        
        # Visualize clusters
        self.clustering_model.visualize_clusters(X, mapped_labels, feature_cols)
        
        # Get statistics
        stats = self.clustering_model.get_cluster_statistics(X, mapped_labels, feature_cols)
        
        # Save model
        self.clustering_model.save_model()
        
        return student_features, metrics
    
    def run_question_targeting(self, student_features):
        """Step 3: Demonstrate targeted question delivery"""
        print("\n" + "="*80)
        print("STEP 3: TARGETED QUESTION TRIGGERING")
        print("="*80)
        
        # Initialize question module
        self.question_module = QuestionTriggeringModule()
        
        # Create sample question bank
        questions = self.question_module.create_sample_question_bank()
        
        # Get question statistics
        self.question_module.get_question_statistics()
        
        # Prepare student data for targeting
        student_clusters = dict(zip(
            student_features['Student Name'],
            student_features['cluster_id']
        ))
        
        student_accuracy = dict(zip(
            student_features['Student Name'],
            student_features['Is_Correct_Binary']
        ))
        
        # Trigger questions for session
        print("\nTriggering questions for students...")
        triggered_questions = self.question_module.trigger_questions_for_session(
            student_clusters=student_clusters,
            student_accuracy=student_accuracy,
            num_questions=2
        )
        
        # Show sample triggered questions
        print("\nSample Triggered Questions:")
        sample_students = list(triggered_questions.keys())[:3]
        for student in sample_students:
            questions = triggered_questions[student]
            print(f"\n{student} (Cluster: {student_clusters[student]} - "
                  f"{self.clustering_model.cluster_labels[student_clusters[student]]}, "
                  f"Accuracy: {student_accuracy[student]:.2f}):")
            for i, q in enumerate(questions, 1):
                print(f"  Q{i}: {q['question_text'][:70]}...")
                print(f"      Difficulty: {q['difficulty']}")
        
        # Evaluate targeting accuracy
        targeting_metrics = self.question_module.evaluate_targeting_accuracy(
            triggered_questions, student_clusters
        )
        
        return triggered_questions, targeting_metrics
    
    def run_feedback_generation(self, student_features):
        """Step 4: Generate personalized feedback"""
        print("\n" + "="*80)
        print("STEP 4: PERSONALIZED FEEDBACK GENERATION")
        print("="*80)
        
        # Initialize feedback generator
        self.feedback_generator = FeedbackGenerator()
        
        # Prepare student data for feedback
        feedback_data = student_features[['Student Name', 'cluster_id', 
                                         'Is_Correct_Binary', 'Response Time (sec)']].copy()
        feedback_data.columns = ['Student Name', 'cluster_id', 'accuracy', 'response_time']
        
        # Add network quality from original data
        network_quality = self.preprocessor.df.groupby('Student Name')['Network Quality'].first().reset_index()
        feedback_data = feedback_data.merge(network_quality, on='Student Name', how='left')
        feedback_data.columns = ['Student Name', 'cluster_id', 'accuracy', 
                                'response_time', 'network_quality']
        
        # Generate batch feedback
        feedback_data_with_feedback = self.feedback_generator.generate_batch_feedback(feedback_data)
        
        # Display sample feedback
        print("\nGenerated Feedback (Sample):")
        print("="*80)
        sample_size = min(5, len(feedback_data_with_feedback))
        for idx, row in feedback_data_with_feedback.head(sample_size).iterrows():
            cluster_name = self.clustering_model.cluster_labels[row['cluster_id']]
            print(f"\n{row['Student Name']}:")
            print(f"  Cluster: {cluster_name}")
            print(f"  Accuracy: {row['accuracy']:.2f}")
            print(f"  Network: {row['network_quality']}")
            print(f"  Feedback: {row['feedback']}")
        
        # Evaluate feedback relevance
        print("\n" + "="*80)
        print("Feedback Relevance Evaluation (Sample)")
        print("="*80)
        
        relevance_scores = []
        for idx, row in feedback_data_with_feedback.head(sample_size).iterrows():
            evaluation = self.feedback_generator.evaluate_feedback_relevance(
                row['feedback'], row['cluster_id'], row['accuracy']
            )
            relevance_scores.append(evaluation['relevance_score'])
            print(f"\n{row['Student Name']}: Score = {evaluation['relevance_score']:.2f}, "
                  f"Relevant = {evaluation['is_relevant']}")
        
        avg_relevance = np.mean(relevance_scores)
        print(f"\nAverage Relevance Score: {avg_relevance:.2f}")
        
        return feedback_data_with_feedback
    
    def generate_summary_report(self, student_features, clustering_metrics, 
                               targeting_metrics, feedback_data):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print(" "*25 + "SYSTEM SUMMARY REPORT")
        print("="*80)
        
        # Dataset summary
        print("\n1. DATASET SUMMARY")
        print("-" * 80)
        print(f"   Total Records: {len(self.preprocessor.df)}")
        print(f"   Total Students: {len(student_features)}")
        print(f"   Total Quizzes: {self.preprocessor.df['Quiz#'].nunique()}")
        print(f"   Total Questions: {self.preprocessor.df['Question ID'].nunique()}")
        
        # Clustering summary
        print("\n2. CLUSTERING RESULTS")
        print("-" * 80)
        cluster_dist = student_features['cluster_name'].value_counts()
        for cluster_name, count in cluster_dist.items():
            percentage = (count / len(student_features)) * 100
            print(f"   {cluster_name} Students: {count} ({percentage:.1f}%)")
        
        print(f"\n   Silhouette Score: {clustering_metrics['Silhouette Score']:.4f}")
        print(f"   Davies-Bouldin Index: {clustering_metrics['Davies-Bouldin Index']:.4f}")
        
        # Question targeting summary
        print("\n3. QUESTION TARGETING")
        print("-" * 80)
        print(f"   Total Questions Triggered: {targeting_metrics['Total Questions Triggered']}")
        print(f"   Targeting Accuracy: {targeting_metrics['Percentage']}")
        
        # Feedback generation summary
        print("\n4. FEEDBACK GENERATION")
        print("-" * 80)
        print(f"   Total Feedback Generated: {len(feedback_data)}")
        feedback_lengths = feedback_data['feedback'].apply(len)
        print(f"   Average Feedback Length: {feedback_lengths.mean():.0f} characters")
        
        # Engagement insights
        print("\n5. KEY INSIGHTS")
        print("-" * 80)
        
        passive_students = student_features[student_features['cluster_id'] == 0]
        if len(passive_students) > 0:
            print(f"   • {len(passive_students)} students need attention (Passive cluster)")
            print(f"     - Average accuracy: {passive_students['Is_Correct_Binary'].mean():.2f}")
        
        active_students = student_features[student_features['cluster_id'] == 2]
        if len(active_students) > 0:
            print(f"   • {len(active_students)} students are highly engaged (Active cluster)")
            print(f"     - Average accuracy: {active_students['Is_Correct_Binary'].mean():.2f}")
        
        # Network quality impact
        network_issues = self.preprocessor.df[
            self.preprocessor.df['Network Quality'] == 'Poor'
        ]['Student Name'].nunique()
        print(f"   • {network_issues} students experiencing network issues")
        
        print("\n" + "="*80)
        print(" "*20 + "PIPELINE EXECUTION COMPLETED")
        print("="*80)
    
    def run_full_pipeline(self):
        """Execute the complete pipeline"""
        try:
            # Step 1: Data Preprocessing
            df = self.run_data_preprocessing()
            
            # Step 2: Clustering
            student_features, clustering_metrics = self.run_clustering()
            
            # Step 3: Question Targeting
            triggered_questions, targeting_metrics = self.run_question_targeting(student_features)
            
            # Step 4: Feedback Generation
            feedback_data = self.run_feedback_generation(student_features)
            
            # Generate Summary Report
            self.generate_summary_report(
                student_features, clustering_metrics, 
                targeting_metrics, feedback_data
            )
            
            # Save results
            self.save_results(student_features, feedback_data)
            
            return {
                'student_features': student_features,
                'clustering_metrics': clustering_metrics,
                'triggered_questions': triggered_questions,
                'targeting_metrics': targeting_metrics,
                'feedback_data': feedback_data
            }
            
        except Exception as e:
            print(f"\nError in pipeline execution: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_results(self, student_features, feedback_data):
        """Save results to CSV files"""
        output_dir = '../data/processed'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save student clusters
        student_features.to_csv(f'{output_dir}/student_clusters.csv', index=False)
        print(f"\nStudent clusters saved to {output_dir}/student_clusters.csv")
        
        # Save feedback
        feedback_data.to_csv(f'{output_dir}/student_feedback.csv', index=False)
        print(f"Student feedback saved to {output_dir}/student_feedback.csv")


if __name__ == "__main__":
    # Initialize and run pipeline
    pipeline = StudentEngagementPipeline(data_path='../Merge.csv')
    
    # Run full pipeline
    results = pipeline.run_full_pipeline()
    
    if results:
        print("\n" + "="*80)
        print("Pipeline execution successful!")
        print("Check the 'data/processed' directory for output files and visualizations.")
        print("="*80)
