# 🛒 Online Shopper Intention Prediction

This project investigates the prediction of online shopping purchase intention using session-level behavioral features from an e-commerce dataset. We evaluate five machine learning models—including Logistic Regression, SVM, Random Forest, XGBoost, and Deep Neural Network—under a unified pipeline with class imbalance handling and performance visualization.

## 📂 Project Structure

```
Project/
├── data/                      # Original & preprocessed datasets
│   ├── online_shoppers_intention.csv
│   ├── preprocessed_data.csv
├── models/                    # Saved model files (.pkl / .h5)
├── notebooks/                 # Development notebooks
│   ├── 01_preprocessing.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_smote_ensemble_models.ipynb
│   ├── 04_dnn_model.ipynb
│   └── 05_evaluation.ipynb
├── results/                   # Output charts and evaluation CSVs
│   ├── *.png, evaluation_summary.csv
├── src/                       # Reproducible pipeline scripts
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt           # Python dependencies
└── README.md
```

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/online-shopper-intent-prediction.git
cd online-shopper-intent-prediction
```

### 2. Set up the environment

You can use `venv` or `conda`. Below is `venv`:

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Reproduce the pipeline

#### Preprocess the data
```bash
python src/preprocess.py \
  --input data/online_shoppers_intention.csv \
  --output data/preprocessed_data.csv
```

#### Train all models
```bash
python src/train.py --models all
```

#### Evaluate and visualize results
```bash
python src/evaluate.py --models all
```

All models and figures will be saved in `models/` and `results/`.

## 📊 Models Included

| Model              | Notes                                |
|-------------------|---------------------------------------|
| Logistic Regression | Baseline with class weighting        |
| SVM               | Linear + RBF kernels, class weighting |
| Random Forest     | SMOTE + tuned ensemble                |
| XGBoost           | SMOTE + log-loss + regularization     |
| Deep Neural Net   | Dropout, early stopping, architecture search |

## 📈 Output Samples

- `results/evaluation_summary.csv`: Summary of F1, precision, recall, AUC
- `results/*.png`: Confusion matrices, ROC curves, metric barplots, gap analysis

## 🧠 Highlights

- Imbalance addressed with both `class_weight` and SMOTE
- Models compared on generalization gap (train vs. test)
- DNN performance benchmarked across 5 architectures
- XGBoost achieved the best overall results (AUC = 0.926)

## 🚀 Future Extensions

- Apply LSTM for sequential clickstream modeling
- Integrate contextual features and user personalization
- Use real-time prediction for marketing automation & recommendation

## 📄 License

MIT License

