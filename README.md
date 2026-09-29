# Employee Attrition Prediction

Predicting whether an employee will leave the company, by benchmarking five classical ML classifiers on the same preprocessing pipeline and evaluation split.

## Approach

1. **Cleaning** – drop rows with missing values and the `Employee ID` column.
2. **Encoding** – label-encode every categorical column.
3. **Scaling** – standardize numeric features with `StandardScaler`.
4. **Split** – 80/20 train/test (`random_state=42`), about 14.9K records in total and 2,980 in the test set, with a roughly balanced target (1,391 stayed / 1,589 left).
5. **Models** – Logistic Regression, Decision Tree, Random Forest (100 trees), SVM (RBF), KNN (k=5).
6. **Evaluation** – accuracy, weighted precision/recall/F1, a per-class classification report and a confusion matrix for every model.

## Results (test set)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Random Forest** | **0.729** | **0.730** | **0.729** | **0.729** |
| Support Vector Machine | 0.716 | 0.718 | 0.716 | 0.716 |
| Logistic Regression | 0.710 | 0.710 | 0.710 | 0.710 |
| Decision Tree | 0.647 | 0.648 | 0.647 | 0.647 |
| K-Nearest Neighbors | ~0.65 | ~0.65 | ~0.65 | ~0.65 |

Random Forest performed best. The ensemble beat the single Decision Tree by about 8 points, which shows how much variance reduction matters on this tabular data.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook employee_attrition_classification.ipynb
```

## Dataset

This project uses the [Employee Attrition Classification Dataset](https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset) (Kaggle, stealthtechnologies), a **synthetic** HR dataset. It uses the dataset's `test.csv` split (about 14.9K records, with an `Attrition` target of Stayed/Left). Download `test.csv` from Kaggle and place it next to the notebook. The dataset isn't included in this repo.

## Next steps

- Cross-validation and hyperparameter tuning (`GridSearchCV`) for Random Forest and SVM
- Gradient boosting (XGBoost / LightGBM)
- Feature importance / SHAP to explain *why* employees leave
- One-hot encoding for nominal features instead of label encoding
- Fit the scaler on the training split only, to avoid test-set leakage
- Train on the dataset's larger `train.csv` and evaluate on `test.csv`

## Tech

Python · pandas · scikit-learn · seaborn · matplotlib
