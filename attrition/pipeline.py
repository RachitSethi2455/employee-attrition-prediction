"""Preprocessing, model zoo and metrics."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .data import NOMINAL, NUMERIC, ORDINAL

SEED = 42


def build_preprocess(nominal=None):
    """Ordinal-encode ordered categories, one-hot the nominal ones, scale everything.

    ``nominal`` lets an ablation drop nominal columns (e.g. sensitive attributes).
    """
    nominal = list(NOMINAL) if nominal is None else list(nominal)
    return ColumnTransformer([
        ("ord", Pipeline([("enc", OrdinalEncoder(categories=list(ORDINAL.values()))),
                          ("scale", StandardScaler())]), list(ORDINAL)),
        ("nom", OneHotEncoder(categories=[NOMINAL[c] for c in nominal], drop="if_binary",
                              handle_unknown="ignore", sparse_output=False), nominal),
        ("num", StandardScaler(), NUMERIC),
    ], verbose_feature_names_out=False)


def model_zoo(seed=SEED):
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
        "SVM (RBF)": SVC(random_state=seed),
        "KNN (k=15)": KNeighborsClassifier(n_neighbors=15),
        "Hist. Gradient Boosting": HistGradientBoostingClassifier(random_state=seed),
    }


def make_pipeline(clf, nominal=None):
    return Pipeline([("prep", build_preprocess(nominal)), ("clf", clf)])


def scores(model, X):
    """Probability of leaving where available, otherwise the decision function."""
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)


def metrics(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_score),
        "Precision (Left)": precision_score(y_true, y_pred),
        "Recall (Left)": recall_score(y_true, y_pred),
        "F1 (Left)": f1_score(y_true, y_pred),
    }
