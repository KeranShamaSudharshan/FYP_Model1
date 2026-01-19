"""
Targeted Question Triggering Module
Delivers appropriate questions based on student engagement clusters
"""

import numpy as np
import pandas as pd
import random
from typing import List, Dict, Optional


class QuestionTriggeringModule:
    """Handles targeted question delivery based on student clusters"""
    
    def __init__(self, random_state=42):
        """
        Initialize the question triggering module
        
        Args:
            random_state: Random seed for reproducible question selection
        """
        self.question_types = ['cluster_based', 'generic', 'random']
        self.cluster_labels = {0: 'Passive', 1: 'Moderate', 2: 'Active'}
        self.random_state = random_state
        
        # Question difficulty mapping based on cluster
        self.difficulty_map = {
            0: 'Easy',      # Passive students get easy questions
            1: 'Medium',    # Moderate students get medium questions
            2: 'Hard'       # Active students get challenging questions
        }
        
    def load_questions(self, questions_df: pd.DataFrame):
        """
        Load question bank from DataFrame
        
        Args:
            questions_df: DataFrame containing questions with columns:
                         - Question, Difficulty, Category, etc.
        """
        self.questions = questions_df
        print(f"Loaded {len(self.questions)} questions")
        
    def create_sample_question_bank(self):
        """
        Create a sample question bank for demonstration
        Based on the Merge.csv dataset structure
        
        Returns:
            DataFrame with sample questions
        """
        questions = []
        
        # Sample questions based on different difficulty levels
        # Easy questions (for Passive students)
        easy_questions = [
            {
                'question_id': 'q_easy_1',
                'question_text': 'What is the SI unit of force?',
                'difficulty': 'Easy',
                'category': 'Physics',
                'cluster_target': 0,
                'options': ['Newton', 'Joule', 'Watt', 'Pascal'],
                'correct_answer': 0
            },
            {
                'question_id': 'q_easy_2',
                'question_text': 'What is the acceleration due to gravity?',
                'difficulty': 'Easy',
                'category': 'Physics',
                'cluster_target': 0,
                'options': ['9.8 m/s²', '10 m/s²', '8.9 m/s²', '11 m/s²'],
                'correct_answer': 0
            },
            {
                'question_id': 'q_easy_3',
                'question_text': 'What is speed?',
                'difficulty': 'Easy',
                'category': 'Physics',
                'cluster_target': 0,
                'options': ['Distance/Time', 'Time/Distance', 'Force/Mass', 'Mass/Force'],
                'correct_answer': 0
            }
        ]
        
        # Medium questions (for Moderate students)
        medium_questions = [
            {
                'question_id': 'q_medium_1',
                'question_text': 'A car accelerates from rest to 30 m/s in 10 seconds. What is the acceleration?',
                'difficulty': 'Medium',
                'category': 'Physics',
                'cluster_target': 1,
                'options': ['3 m/s²', '2 m/s²', '4 m/s²', '5 m/s²'],
                'correct_answer': 0
            },
            {
                'question_id': 'q_medium_2',
                'question_text': 'What is the kinetic energy of a 2 kg object moving at 5 m/s?',
                'difficulty': 'Medium',
                'category': 'Physics',
                'cluster_target': 1,
                'options': ['25 J', '50 J', '10 J', '20 J'],
                'correct_answer': 0
            },
            {
                'question_id': 'q_medium_3',
                'question_text': 'Calculate the momentum of a 10 kg object moving at 4 m/s.',
                'difficulty': 'Medium',
                'category': 'Physics',
                'cluster_target': 1,
                'options': ['40 kg⋅m/s', '20 kg⋅m/s', '30 kg⋅m/s', '50 kg⋅m/s'],
                'correct_answer': 0
            }
        ]
        
        # Hard questions (for Active students)
        hard_questions = [
            {
                'question_id': 'q_hard_1',
                'question_text': 'A projectile is launched at 45° with velocity 20 m/s. What is the maximum height?',
                'difficulty': 'Hard',
                'category': 'Physics',
                'cluster_target': 2,
                'options': ['10.2 m', '15.5 m', '20.4 m', '5.1 m'],
                'correct_answer': 0
            },
            {
                'question_id': 'q_hard_2',
                'question_text': 'Two bodies of mass 3kg and 2kg collide elastically. What is the final velocity ratio?',
                'difficulty': 'Hard',
                'category': 'Physics',
                'cluster_target': 2,
                'options': ['3:2', '2:3', '1:1', '4:3'],
                'correct_answer': 1
            },
            {
                'question_id': 'q_hard_3',
                'question_text': 'Calculate work done when a 5N force moves an object 10m at 60° angle.',
                'difficulty': 'Hard',
                'category': 'Physics',
                'cluster_target': 2,
                'options': ['25 J', '43.3 J', '50 J', '30 J'],
                'correct_answer': 0
            }
        ]
        
        # Generic questions (for all students)
        generic_questions = [
            {
                'question_id': 'q_generic_1',
                'question_text': 'What is Newton\'s First Law of Motion?',
                'difficulty': 'Medium',
                'category': 'Physics',
                'cluster_target': -1,  # -1 indicates generic
                'options': ['Law of Inertia', 'F=ma', 'Action-Reaction', 'Conservation of Energy'],
                'correct_answer': 0
            },
            {
                'question_id': 'q_generic_2',
                'question_text': 'What is the unit of power?',
                'difficulty': 'Medium',
                'category': 'Physics',
                'cluster_target': -1,
                'options': ['Watt', 'Joule', 'Newton', 'Pascal'],
                'correct_answer': 0
            }
        ]
        
        # Combine all questions
        all_questions = easy_questions + medium_questions + hard_questions + generic_questions
        
        self.questions = pd.DataFrame(all_questions)
        print(f"Created sample question bank with {len(self.questions)} questions")
        
        return self.questions
    
    def get_cluster_based_question(self, cluster_id: int) -> Optional[Dict]:
        """
        Get a question targeted for a specific cluster
        
        Args:
            cluster_id: Cluster ID (0=Passive, 1=Moderate, 2=Active)
            
        Returns:
            Question dictionary or None
        """
        # Filter questions for the target cluster
        cluster_questions = self.questions[
            self.questions['cluster_target'] == cluster_id
        ]
        
        if len(cluster_questions) == 0:
            print(f"No questions found for cluster {cluster_id}")
            return None
        
        # Select a random question from the filtered set
        question = cluster_questions.sample(n=1, random_state=self.random_state).iloc[0]
        
        return question.to_dict()
    
    def get_generic_question(self) -> Optional[Dict]:
        """
        Get a generic question suitable for all students
        
        Returns:
            Question dictionary or None
        """
        # Filter generic questions (cluster_target = -1)
        generic_questions = self.questions[
            self.questions['cluster_target'] == -1
        ]
        
        if len(generic_questions) == 0:
            print("No generic questions found")
            return None
        
        # Select a random question
        question = generic_questions.sample(n=1, random_state=self.random_state).iloc[0]
        
        return question.to_dict()
    
    def get_random_question(self) -> Optional[Dict]:
        """
        Get a completely random question
        
        Returns:
            Question dictionary or None
        """
        if len(self.questions) == 0:
            print("No questions available")
            return None
        
        # Select any random question
        question = self.questions.sample(n=1, random_state=self.random_state).iloc[0]
        
        return question.to_dict()
    
    def get_adaptive_question(self, cluster_id: int, 
                            previous_accuracy: float,
                            question_type: str = 'cluster_based') -> Optional[Dict]:
        """
        Get an adaptive question based on cluster and previous performance
        
        Args:
            cluster_id: Student's current cluster ID
            previous_accuracy: Student's recent accuracy (0.0 to 1.0)
            question_type: Type of question ('cluster_based', 'generic', 'random')
            
        Returns:
            Question dictionary or None
        """
        # Adjust difficulty based on previous accuracy
        if previous_accuracy >= 0.8:  # High accuracy
            adjusted_cluster = min(cluster_id + 1, 2)  # Try harder questions
        elif previous_accuracy <= 0.4:  # Low accuracy
            adjusted_cluster = max(cluster_id - 1, 0)  # Try easier questions
        else:
            adjusted_cluster = cluster_id  # Keep same difficulty
        
        # Get question based on type
        if question_type == 'cluster_based':
            question = self.get_cluster_based_question(adjusted_cluster)
        elif question_type == 'generic':
            question = self.get_generic_question()
        elif question_type == 'random':
            question = self.get_random_question()
        else:
            raise ValueError(f"Unknown question type: {question_type}")
        
        if question:
            question['adaptive_cluster'] = adjusted_cluster
            question['original_cluster'] = cluster_id
            question['based_on_accuracy'] = previous_accuracy
        
        return question
    
    def trigger_questions_for_session(self, student_clusters: Dict[str, int],
                                     student_accuracy: Optional[Dict[str, float]] = None,
                                     num_questions: int = 1) -> Dict[str, List[Dict]]:
        """
        Trigger questions for all students in a session
        
        Args:
            student_clusters: Dictionary mapping student names to cluster IDs
            student_accuracy: Optional dictionary of student accuracies
            num_questions: Number of questions to trigger per student
            
        Returns:
            Dictionary mapping student names to list of questions
        """
        triggered_questions = {}
        
        for student_name, cluster_id in student_clusters.items():
            student_questions = []
            
            # Get student's accuracy if available
            accuracy = student_accuracy.get(student_name, 0.5) if student_accuracy else 0.5
            
            for _ in range(num_questions):
                # Get adaptive question
                question = self.get_adaptive_question(
                    cluster_id=cluster_id,
                    previous_accuracy=accuracy,
                    question_type='cluster_based'
                )
                
                if question:
                    student_questions.append(question)
            
            triggered_questions[student_name] = student_questions
        
        return triggered_questions
    
    def evaluate_targeting_accuracy(self, triggered_questions: Dict,
                                   student_clusters: Dict[str, int]) -> Dict:
        """
        Evaluate the accuracy of question targeting
        
        Args:
            triggered_questions: Dictionary of triggered questions
            student_clusters: Dictionary of student cluster IDs
            
        Returns:
            Dictionary with evaluation metrics
        """
        total_questions = 0
        correctly_targeted = 0
        
        for student_name, questions in triggered_questions.items():
            cluster_id = student_clusters[student_name]
            
            for question in questions:
                total_questions += 1
                
                # Check if question target matches student cluster
                if question.get('cluster_target', -1) == cluster_id:
                    correctly_targeted += 1
                elif question.get('cluster_target', -1) == -1:  # Generic questions
                    correctly_targeted += 0.5  # Partial credit
        
        accuracy = correctly_targeted / total_questions if total_questions > 0 else 0
        
        metrics = {
            'Total Questions Triggered': total_questions,
            'Correctly Targeted': correctly_targeted,
            'Targeting Accuracy': accuracy,
            'Percentage': f"{accuracy * 100:.2f}%"
        }
        
        print("\nQuestion Targeting Evaluation:")
        print("="*60)
        for metric, value in metrics.items():
            print(f"{metric:30s}: {value}")
        print("="*60)
        
        return metrics
    
    def get_question_statistics(self) -> pd.DataFrame:
        """
        Get statistics about the question bank
        
        Returns:
            DataFrame with question statistics
        """
        stats = []
        
        # Overall statistics
        stats.append({
            'Category': 'Total',
            'Count': len(self.questions),
            'Percentage': '100%'
        })
        
        # By difficulty
        difficulty_counts = self.questions['difficulty'].value_counts()
        for difficulty, count in difficulty_counts.items():
            stats.append({
                'Category': f'Difficulty: {difficulty}',
                'Count': count,
                'Percentage': f"{count/len(self.questions)*100:.2f}%"
            })
        
        # By cluster target
        cluster_counts = self.questions['cluster_target'].value_counts()
        cluster_names = {0: 'Passive', 1: 'Moderate', 2: 'Active', -1: 'Generic'}
        for cluster_id, count in cluster_counts.items():
            stats.append({
                'Category': f'Target: {cluster_names.get(cluster_id, "Unknown")}',
                'Count': count,
                'Percentage': f"{count/len(self.questions)*100:.2f}%"
            })
        
        stats_df = pd.DataFrame(stats)
        
        print("\nQuestion Bank Statistics:")
        print("="*60)
        print(stats_df.to_string(index=False))
        print("="*60)
        
        return stats_df


if __name__ == "__main__":
    # Test the question triggering module
    
    # Initialize module
    qt_module = QuestionTriggeringModule()
    
    # Create sample question bank
    questions = qt_module.create_sample_question_bank()
    
    # Get statistics
    qt_module.get_question_statistics()
    
    # Test question retrieval
    print("\n" + "="*60)
    print("Testing Question Retrieval")
    print("="*60)
    
    # Get cluster-based questions
    for cluster_id in [0, 1, 2]:
        print(f"\nCluster {cluster_id} ({qt_module.cluster_labels[cluster_id]}):")
        question = qt_module.get_cluster_based_question(cluster_id)
        if question:
            print(f"  Question: {question['question_text']}")
            print(f"  Difficulty: {question['difficulty']}")
    
    # Get generic question
    print("\nGeneric Question:")
    question = qt_module.get_generic_question()
    if question:
        print(f"  Question: {question['question_text']}")
    
    # Test adaptive questioning
    print("\n" + "="*60)
    print("Testing Adaptive Question Selection")
    print("="*60)
    
    # Simulate student clusters
    student_clusters = {
        'Student A': 0,  # Passive
        'Student B': 1,  # Moderate
        'Student C': 2,  # Active
    }
    
    student_accuracy = {
        'Student A': 0.3,  # Low accuracy
        'Student B': 0.7,  # Good accuracy
        'Student C': 0.9,  # High accuracy
    }
    
    # Trigger questions
    triggered = qt_module.trigger_questions_for_session(
        student_clusters=student_clusters,
        student_accuracy=student_accuracy,
        num_questions=2
    )
    
    for student, questions in triggered.items():
        print(f"\n{student} (Cluster: {student_clusters[student]}, Accuracy: {student_accuracy[student]}):")
        for i, q in enumerate(questions, 1):
            print(f"  Q{i}: {q['question_text'][:60]}...")
            print(f"      Difficulty: {q['difficulty']}, Target Cluster: {q.get('adaptive_cluster', 'N/A')}")
    
    # Evaluate targeting accuracy
    qt_module.evaluate_targeting_accuracy(triggered, student_clusters)
