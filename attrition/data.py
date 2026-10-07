"""Loading the Kaggle employee-attrition dataset and the feature schema."""
import shutil
from pathlib import Path

import pandas as pd

KAGGLE_DATASET = "stealthtechnologies/employee-attrition-dataset"
TARGET = "Attrition"
ID_COLUMN = "Employee ID"
POSITIVE_LABEL = "Left"

# Ordered categories, listed from lowest to highest.
ORDINAL = {
    "Work-Life Balance": ["Poor", "Fair", "Good", "Excellent"],
    "Job Satisfaction": ["Low", "Medium", "High", "Very High"],
    "Performance Rating": ["Low", "Below Average", "Average", "High"],
    "Education Level": ["High School", "Associate Degree", "Bachelor’s Degree", "Master’s Degree", "PhD"],
    "Job Level": ["Entry", "Mid", "Senior"],
    "Company Size": ["Small", "Medium", "Large"],
    "Company Reputation": ["Poor", "Fair", "Good", "Excellent"],
    "Employee Recognition": ["Low", "Medium", "High", "Very High"],
}
NOMINAL = {
    "Gender": ["Female", "Male"],
    "Job Role": ["Education", "Finance", "Healthcare", "Media", "Technology"],
    "Overtime": ["No", "Yes"],
    "Marital Status": ["Divorced", "Married", "Single"],
    "Remote Work": ["No", "Yes"],
    "Leadership Opportunities": ["No", "Yes"],
    "Innovation Opportunities": ["No", "Yes"],
}
NUMERIC = [
    "Age", "Years at Company", "Monthly Income", "Number of Promotions",
    "Distance from Home", "Number of Dependents", "Company Tenure",
]
FEATURES = NUMERIC + list(ORDINAL) + list(NOMINAL)


def load_data(data_dir="data"):
    """Return (train_df, test_df), downloading them from Kaggle on first use."""
    data_dir = Path(data_dir)
    if not (data_dir / "train.csv").exists():
        import kagglehub
        src = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        data_dir.mkdir(parents=True, exist_ok=True)
        for f in ("train.csv", "test.csv"):
            shutil.copy(src / f, data_dir / f)
    return pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "test.csv")


def split_xy(df):
    """Features in a fixed column order, and the target with Left = 1."""
    return df[FEATURES], (df[TARGET] == POSITIVE_LABEL).astype(int)
