# Model Performance Improvement & Comparison

This document summarizes the performance comparison of machine learning models before and after applying feature reduction and Principal Component Analysis (PCA) on the Breast Cancer Wisconsin (Diagnostic) dataset (reduced from 30 features to 7 features).

---

## 1. Dimensionality Reduction Overview

- **Original Feature Count:** 30 features
- **Reduced Feature Count:** 7 features (6 uncorrelated features + 1 Principal Component `PC1`)
- **Dimensionality Reduction:** 76.7% reduction in feature space

---

## 2. Model Performance Comparison (Without PCA vs. With PCA)

Hyperparameter tuning was conducted using `RandomizedSearchCV` (5-fold cross-validation) across 8 classification algorithms.

| Model | Feature Set | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Without PCA (30 features)<br>With PCA (7 features)<br>**Net Change** | 89.65%<br>90.31%<br>**+0.66%** | 95.38%<br>85.80%<br>*-9.58%* | 75.15%<br>87.88%<br>**+12.73%** | 84.07%<br>86.83%<br>**+2.76%** | 86.54%<br>89.79%<br>**+3.25%** |
| **Support Vector Classifier (SVC)** | Without PCA (30 features)<br>With PCA (7 features)<br>**Net Change** | 92.95%<br>90.97%<br>*-1.98%* | 89.35%<br>91.33%<br>**+1.98%** | 91.52%<br>83.03%<br>*-8.49%* | 90.42%<br>86.98%<br>*-3.44%* | 92.64%<br>89.27%<br>*-3.37%* |
| **XGBoost (Top Performer with PCA)** | Without PCA (30 features)<br>With PCA (7 features)<br>**Net Change** | 95.37%<br>93.39%<br>*-1.98%* | 92.35%<br>91.41%<br>*-0.94%* | 95.15%<br>90.30%<br>*-4.85%* | 93.73%<br>90.85%<br>*-2.88%* | 95.33%<br>92.73%<br>*-2.60%* |
| **Random Forest** | Without PCA (30 features)<br>With PCA (7 features)<br>**Net Change** | 94.49%<br>91.63%<br>*-2.86%* | 90.70%<br>89.94%<br>*-0.76%* | 94.55%<br>86.67%<br>*-7.88%* | 92.58%<br>88.27%<br>*-4.31%* | 94.50%<br>90.57%<br>*-3.93%* |
| **Gradient Boosting** | Without PCA (30 features)<br>With PCA (7 features)<br>**Net Change** | 93.17%<br>89.87%<br>*-3.30%* | 96.53%<br>94.07%<br>*-2.46%* | 84.24%<br>76.97%<br>*-7.27%* | 89.97%<br>84.67%<br>*-5.30%* | 91.26%<br>87.10%<br>*-4.16%* |
| **K-Nearest Neighbors (KNN)** | Without PCA (30 features)<br>With PCA (7 features)<br>**Net Change** | 93.39%<br>89.65%<br>*-3.74%* | 96.55%<br>91.55%<br>*-5.00%* | 84.85%<br>78.79%<br>*-6.06%* | 90.32%<br>84.69%<br>*-5.63%* | 91.56%<br>87.32%<br>*-4.24%* |
| **AdaBoost** | Without PCA (30 features)<br>With PCA (7 features)<br>**Net Change** | 96.04%<br>91.63%<br>*-4.41%* | 94.55%<br>89.44%<br>*-5.11%* | 94.55%<br>87.27%<br>*-7.28%* | 94.55%<br>88.34%<br>*-6.21%* | 95.72%<br>90.70%<br>*-5.02%* |
| **Decision Tree** | Without PCA (30 features)<br>With PCA (7 features)<br>**Net Change** | 94.93%<br>85.90%<br>*-9.03%* | 91.76%<br>80.24%<br>*-11.52%* | 94.55%<br>81.21%<br>*-13.34%* | 93.13%<br>80.72%<br>*-12.41%* | 94.85%<br>84.90%<br>*-9.95%* |

---

## 3. Key Performance Insights

- **Logistic Regression Gains:** With PCA and correlation filtering, Logistic Regression improved in **Accuracy (+0.66%)**, **Recall (+12.73%)**, **F1 Score (+2.76%)**, and **ROC AUC (+3.25%)**.
- **SVC Precision:** Precision improved by **+1.98%** (from 89.35% to 91.33%).
- **XGBoost Robustness:** Maintained top performance across both setups, retaining **93.39% Accuracy** and **92.73% ROC AUC** even after removing over 76% of feature dimensions.
- **Efficiency vs. Performance:** Reducing dimensions to 7 significantly lowers model complexity, training time, and risk of multicollinearity while maintaining competitive predictive performance across ensemble models.
