# CMRL
Causal Metapath-guided Heterogeneous Graph Representation Learning for disease triple-wise association mining

Complete code and data are coming

data_process.py - Data Processing Module
Data Flow:
Data Preparation (data_process.py)
Raw data → Feature matrices → Negative samples → Train/Test splits

main.py - Main Execution Module
Experiment Execution (main.py)
Load data → Construct graph → Train model → Evaluate → Save results

model.py - Neural Network Model Module/train.py - Training Module
Model Training (train.py + model.py)
Input features → Bayesian encoding → Graph propagation → Information bottleneck → Prediction

utils.py - Utilities Module
Evaluation (utils.py)
Predictions → Ranking → Hits@n/NDCG@n/MRR calculation
