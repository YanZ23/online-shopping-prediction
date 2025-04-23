# Online Shopping Prediction

This project aims to predict whether an online shopping session results in a purchase using the Online Shoppers Purchasing Intention Dataset. We experiment with several machine learning models and address class imbalance using SMOTE and class weighting. The project provides insights into user behavior that may help improve e-commerce conversion rates.

Project Overview:
- Task: Binary classification (Revenue = True / False)
- Dataset: 12,330 browsing sessions
- Features: 17 attributes (e.g., bounce rates, exit rates, visitor type, weekend, etc.)
- Goal: Identify key factors that influence purchase decisions

Challenges Addressed:
- Class Imbalance: Only ~15.5% of samples are positive (Revenue=True)
- Mixed Features: Numerical and categorical variables
- Robust Evaluation: Focus on F1, Precision, Recall, and ROC-AUC instead of accuracy

Models Evaluated:
- Logistic Regression (Baseline)
- Support Vector Machine (SVM)
- Random Forest
- XGBoost
- Deep Neural Networks (DNN)

Evaluation Metrics:
- F1-Score
- Precision / Recall
- ROC-AUC Score
- Confusion Matrix

Project Structure:
online-shopping-prediction/
│
├── data/
│   └── online_shoppers_intention.csv
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_tree_models.ipynb
│   ├── 04_dnn_model.ipynb
│   ├── 05_smote_weighting.ipynb
│   └── 06_final_evaluation.ipynb
│
├── src/
│   ├── train_models.py
│   ├── evaluation.py
│   └── utils.py
│
├── models/
├── results/
├── requirements.txt
└── README.md

How to Run:
1. Clone the repo and set up environment:

    git clone git@github.com:YanZ23/online-shopping-prediction.git
    cd online-shopping-prediction

    python3 -m venv venv
    source venv/bin/activate      # On Windows: venv\Scripts\activate

    pip install -r requirements.txt

2. Launch Jupyter Notebook and run notebooks in order:

    jupyter notebook

Dependencies:
Listed in requirements.txt. Main packages include:
- pandas
- scikit-learn
- imbalanced-learn
- xgboost
- matplotlib
- seaborn
- jupyter

Dataset Source:
UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset


Author:
Yanrun Zhu
zhu.yanr@northeastern.edu  
GitHub: https://github.com/YanZ23

License:
This project is intended for educational and academic use only.
