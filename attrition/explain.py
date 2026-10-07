"""Per-employee explanations: SHAP values summed back onto the original features."""
import numpy as np
import pandas as pd

from .data import FEATURES, NOMINAL


def _original_feature(encoded_name):
    if encoded_name in FEATURES:
        return encoded_name
    for col in NOMINAL:
        if encoded_name.startswith(col + "_"):
            return col
    raise KeyError(f"cannot map encoded feature {encoded_name!r} to an input column")


def shap_by_feature(model, X):
    """SHAP contributions (log-odds of leaving) per original input feature.

    Returns ``(contributions, base_value)``. ``contributions`` is a DataFrame with one
    row per row of ``X`` and one column per feature in ``X``. For every row,
    ``base_value + contributions.sum(axis=1)`` equals the model's log-odds.
    """
    import shap

    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    Xt = prep.transform(X)
    explainer = shap.TreeExplainer(clf)
    values = np.asarray(explainer.shap_values(Xt))
    if values.ndim == 3:  # some shap versions return one slice per class
        values = values[..., 1]
    encoded = pd.DataFrame(values, columns=prep.get_feature_names_out(), index=X.index)
    grouped = encoded.T.groupby(_original_feature, sort=False).sum().T
    base = np.ravel(explainer.expected_value)[-1]
    return grouped[[c for c in X.columns if c in grouped.columns]], float(base)
