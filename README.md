
# Telecom Churn Predictor

**Churn classification model for telecom customer datasets.  
94.60% Accuracy | 0.8968 AUC | 0.8675 Precision | 0.7423 Recall**

Predicts which customers are likely to leave using a stacked ensemble of four classifiers.  
Built with **real telecom data**, trained with stratified validation, and fully reproducible.  
This repository includes a complete pipeline: feature engineering, model stacking, and evaluation.

> *For context, Charter Communications, the telecom company this was built for, was previously relying on a spaCy-based model that achieved only ~40% accuracy.*
---

## What It Does

This model predicts customer churn for telecom operators.  
It learns from customer usage patterns, billing behavior, service plans, and support interactions.  
Given raw input data, it outputs a churn probability between 0 and 1 for each customer.

The output helps retention teams target at-risk customers before they leave.

---

## Evaluation Metrics 

| Metric    | Value   | Description |
|-----------|---------|-------------|
| Accuracy  | 94.60%  | Overall correct predictions |
| AUC       | 0.8968  | Separation between churners and non-churners |
| Precision | 0.8675  | % of predicted churners that actually churned |
| Recall    | 0.7423  | % of actual churners correctly identified |

Evaluation done on 80/20 stratified train/test split.  
Base learners trained using k-fold CV. Meta-learner trained on out-of-fold predictions.

---

### Model Architecture: Two-Level Stacked Ensemble

The model is structured as a **two-tiered ensemble**, where each layer plays a distinct role in prediction.

#### **Level 1: Base Learners**

At the first level, only one model is used:

- `XGBoostClassifier`

This model learns patterns from customer attributes (usage, billing, service plans, etc.) and produces a churn probability.

#### **Level 2: Meta-Learners**

The predicted churn probability from Level 1 (XGBoost) is **combined with the original feature set**, then used as input to train three meta-learners:

- `LogisticRegression`
- `DecisionTreeClassifier`
- `GaussianNB`

Each of these meta-models learns a slightly different decision boundary based on the XGBoost signal and the original data. These models each output a second-level probability of churn.

These three second-layer probabilities are then **combined via a weighted soft vote**:

- Logistic Regression: 0.4
- Decision Tree: 0.3
- Naive Bayes: 0.3

The result is a final, blended churn probability that reflects multiple modeling assumptions.

This architecture improves generalization and avoids over-reliance on any single model’s biases. It was designed to reproduce and extend the structure proposed in the MDPI telecom churn ensemble study.

---

## Reproducibility > Claims

Publishing performance numbers in a PDF means nothing without **verifiable code**, **data**, and **methodology**.  
Claims like “99.28% accuracy” are scientifically meaningless without reproducibility.

In churn or NBO—where data is tabular, class distributions are imbalanced, and evaluation choices impact every number—**reproducibility is everything**.

GitHub models provide something better than publication in an academic journals: **proof**.

If you can inspect the code, run the pipeline, and trace the model's decisions on real data—  
you’ve got a **testable asset**, not a claim.

---

## Design Snapshot

| Component           | My Model                                                           |
|---------------------|--------------------------------------------------------------------|
| Bucketing Strategy  | Per-feature Sturges-based bin count with equidistant discretizing |
| Ensemble Structure  | Two-stage pipeline: XGB → (LR, DT, NB) → weighted soft vote        |
| Train/Test Split    | 80/20 stratified                                                  |
| Feature Selection   | Original + 12 grouped features (33 total), matching MDPI design   |
| Voting Mechanism    | Weighted soft vote (LR: 0.4, DT: 0.3, NB: 0.3)                     |

---

## Conclusion

The final system is a layered, structured ensemble with strong performance and high transparency. Logistic regression as a meta-learner effectively balances outputs from diverse base classifiers, while quantile-based bucketing and complete categorical encoding ensure that the full information space is available during training.

A 94.6% accuracy and 0.8968 AUC make this implementation a strong benchmark for practical churn prediction. The modular architecture, clean feature processing, and documented evaluation steps support easy replication and extension—whether for production deployment or integration with retention strategy tools.

---

```
├── churn_model/
│   ├── predict.py
│   ├── train.py
│   ├── artifacts/
│   │   ├── xgb_model.joblib
│   │   ├── lr_model.joblib
│   │   ├── dt_model.joblib
│   │   ├── nb_model.joblib
│   │   └── preprocessor.joblib
```

---

## How to Run

```bash
git clone https://github.com/yourusername/telecom-churn-predictor.git
cd telecom-churn-predictor
pip install -r requirements.txt
python train.py
```
