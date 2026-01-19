# Student Engagement Analytics System

## Overview
This project implements a comprehensive Student Engagement Analytics System for video conferencing-based learning environments. The system uses machine learning to monitor student engagement, predict learner clusters, deliver targeted questions, and generate personalized feedback in real-time.

## Project Structure
```
FYP_Model1/
├── src/
│   ├── models/
│   │   ├── clustering_model.py        # K-Means clustering for student engagement
│   │   ├── question_targeting.py      # Targeted question delivery system
│   │   └── feedback_generation.py     # Personalized feedback generator
│   ├── utils/
│   │   └── data_preprocessing.py      # Data loading and preprocessing
│   └── main_pipeline.py               # End-to-end pipeline integration
├── data/
│   └── processed/                     # Output files and visualizations
├── Merge.csv                          # Student engagement dataset
├── TechSnatchers_SDS (5).docx        # System Design Specification document
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Features

### 1. Data Preprocessing
- Loads and cleans the Merge.csv dataset (5099 records, 20 features)
- Handles missing values and performs feature engineering
- Creates derived features for clustering analysis
- Aggregates student-level metrics

### 2. Predictive Learner Clustering Model
- **Algorithm**: K-Means clustering with k-means++ initialization
- **Features**: Response time, network metrics (RTT, Jitter, Stability), answer correctness, engagement level
- **Clusters**: 3 engagement levels (Passive, Moderate, Active)
- **Evaluation Metrics**:
  - Silhouette Score (cluster quality)
  - Davies-Bouldin Index (separation)
  - Calinski-Harabasz Index (variance ratio)
- **Visualizations**: Elbow curve, PCA-based cluster visualization

### 3. Targeted Question Triggering Module
- Delivers questions based on student engagement clusters
- **Question Types**:
  - Cluster-based (difficulty matched to engagement level)
  - Generic (suitable for all students)
  - Random (any question)
- **Adaptive Delivery**: Adjusts difficulty based on previous accuracy
- Sample question bank with 11 questions across 3 difficulty levels

### 4. Personalized Feedback Generation
- Rule-based feedback system with template matching
- Generates feedback based on:
  - Student cluster (Passive/Moderate/Active)
  - Accuracy level (Low/Medium/High)
  - Performance trends (improving/declining)
  - Network quality issues
- Feedback relevance evaluation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KeranShamaSudharshan/FYP_Model1.git
cd FYP_Model1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Complete Pipeline
```bash
cd src
python main_pipeline.py
```

This will execute all four components sequentially:
1. Data preprocessing
2. Clustering model training and evaluation
3. Question targeting demonstration
4. Feedback generation

### Run Individual Components

**Data Preprocessing:**
```bash
cd src/utils
python data_preprocessing.py
```

**Clustering Model:**
```bash
cd src/models
python clustering_model.py
```

**Question Targeting:**
```bash
cd src/models
python question_targeting.py
```

**Feedback Generation:**
```bash
cd src/models
python feedback_generation.py
```

## Output Files

After running the pipeline, the following files will be generated in `data/processed/`:

- `elbow_curve.png` - Elbow method visualization for optimal K
- `cluster_visualization.png` - 2D PCA visualization of student clusters
- `clustering_model.pkl` - Trained K-Means model
- `student_clusters.csv` - Student cluster assignments and features
- `student_feedback.csv` - Generated personalized feedback for each student

## Dataset

The **Merge.csv** dataset contains:
- 5099 quiz response records
- 20 features including:
  - Student information (Name, Email, Admission No)
  - Quiz details (Question, Answer, Correctness)
  - Performance metrics (Response time, Accuracy)
  - Network metrics (RTT, Jitter, Stability, Quality)
  - Engagement level (Active, Moderate, Passive)

## Model Performance

### Clustering Model
- **Number of Clusters**: 3 (Passive, Moderate, Active)
- **Features**: 6 (Response Time, RTT, Jitter, Stability, Correctness, Engagement)
- **Evaluation**: Silhouette Score, Davies-Bouldin Index

### Question Targeting
- **Targeting Accuracy**: Measured by cluster-question difficulty alignment
- **Adaptive Mechanism**: Adjusts based on student accuracy trends

### Feedback Generation
- **Feedback Types**: 9 cluster-accuracy combinations + trend-based + network-based
- **Relevance Score**: Evaluated for appropriateness and encouragement

## System Architecture

Based on the SDS document, the system uses:
- **Architecture**: 3-tier (Web Application, Backend, Database)
- **Development Approach**: Agile (Scrum framework)
- **Database**: MongoDB (document-oriented NoSQL)
- **Integration**: Zoom API for real-time session data

## Research Objectives

1. ✅ Analyze key factors influencing student engagement
2. ✅ Develop predictive learner clustering model
3. ✅ Develop targeted interaction mechanisms
4. ✅ Develop personalized feedback generation model
5. ⏳ Integrate with video conferencing platform (future work)
6. ⏳ Evaluate system performance in live sessions (future work)

## Future Enhancements

1. **LSTM-based Feedback Model**: Implement Seq2Seq LSTM with attention mechanism for more sophisticated feedback generation
2. **Real-time Integration**: Connect with Zoom API for live session monitoring
3. **Web Dashboard**: Build instructor/student dashboards for real-time analytics
4. **Advanced Features**:
   - Session-level analytics
   - Historical trend analysis
   - Multi-session clustering
   - Automated report generation

## Technologies Used

- **Python 3.8+**
- **scikit-learn**: Machine learning (K-Means clustering)
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Data visualization
- **python-docx**: Document reading

## Documentation

For detailed system design, architecture, and implementation details, refer to:
- **TechSnatchers_SDS (5).docx** - Complete System Design Specification

## Team

**Tech Snatchers**
- Supervised by: Dr. K.A.S.H. Kulathilake
- Faculty of Applied Sciences
- Rajarata University of Sri Lanka
- 2020/2021 Batch

## License

This project is part of a Final Year Project (FYP) at Rajarata University of Sri Lanka.

## Contact

For questions or collaboration:
- Repository: https://github.com/KeranShamaSudharshan/FYP_Model1
