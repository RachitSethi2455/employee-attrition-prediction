"""Fast tests on synthetic rows; they need neither the Kaggle data nor network access."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from attrition.data import FEATURES, NOMINAL, NUMERIC, ORDINAL, split_xy
from attrition.explain import shap_by_feature
from attrition.pipeline import build_preprocess, make_pipeline

MODELS = Path(__file__).resolve().parents[1] / "models"


def synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.integers(0, 100, n) for c in NUMERIC})
    for c, cats in {**ORDINAL, **NOMINAL}.items():
        df[c] = rng.choice(cats, n)
    df["Attrition"] = rng.choice(["Stayed", "Left"], n)
    df["Employee ID"] = np.arange(n)
    return df


def test_split_xy_encodes_left_as_positive():
    df = synthetic()
    X, y = split_xy(df)
    assert list(X.columns) == FEATURES
    assert (y == (df["Attrition"] == "Left")).all()


def test_preprocess_output_width():
    X, _ = split_xy(synthetic())
    out = build_preprocess().fit_transform(X)
    one_hot = sum(1 if len(c) == 2 else len(c) for c in NOMINAL.values())  # binary features keep one column
    assert out.shape == (len(X), len(ORDINAL) + one_hot + len(NUMERIC))


def test_ordinal_encoding_respects_real_order():
    X, _ = split_xy(synthetic())
    prep = build_preprocess().fit(X)
    rows = X.head(len(ORDINAL["Work-Life Balance"])).copy()
    rows["Work-Life Balance"] = ORDINAL["Work-Life Balance"]  # Poor, Fair, Good, Excellent
    encoded = pd.DataFrame(prep.transform(rows), columns=prep.get_feature_names_out())["Work-Life Balance"]
    assert encoded.is_monotonic_increasing and encoded.is_unique


def test_unseen_nominal_category_does_not_crash():
    X, _ = split_xy(synthetic())
    prep = build_preprocess().fit(X)
    X.loc[X.index[0], "Job Role"] = "Astronaut"
    assert np.isfinite(prep.transform(X)).all()


def test_shap_contributions_add_up_to_log_odds():
    X, y = split_xy(synthetic(400))
    model = make_pipeline(HistGradientBoostingClassifier(max_iter=30, random_state=0)).fit(X, y)
    contrib, base = shap_by_feature(model, X.head(50))
    assert list(contrib.columns) == FEATURES
    np.testing.assert_allclose(base + contrib.sum(axis=1), model.decision_function(X.head(50)), atol=1e-4)


@pytest.mark.skipif(not (MODELS / "attrition_model.joblib").exists(), reason="run the notebook to create models/")
def test_saved_model_and_card():
    model = joblib.load(MODELS / "attrition_model.joblib")
    card = json.loads((MODELS / "model_card.json").read_text(encoding="utf-8"))
    assert card["features"] == FEATURES
    X, _ = split_xy(synthetic())
    p = model.predict_proba(X)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
    assert card["test_metrics"]["ROC-AUC"] > 0.8


@pytest.mark.skipif(not (MODELS / "attrition_model.joblib").exists(), reason="run the notebook to create models/")
def test_demo_app_prediction():
    import app

    values = {f: (app.card["numeric_ranges"][f][2] if f in NUMERIC else app.card["categorical_defaults"][f]) for f in FEATURES}
    p, band, contrib = app.predict_employee(values)
    assert 0 <= p <= 1 and band in {"Low", "Medium", "High"}
    assert set(contrib.index) == set(FEATURES)
    probs, text, fig = app.run(*[values[f] for f in FEATURES])
    assert abs(sum(probs.values()) - 1) < 1e-9 and "risk" in text
