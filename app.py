"""Gradio demo: enter an employee profile, get the attrition risk and what drives it.

Run locally with ``python app.py`` (after the notebook has saved ``models/``).
"""
import json
from pathlib import Path

import gradio as gr
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from attrition.data import FEATURES, NOMINAL, NUMERIC, ORDINAL
from attrition.explain import shap_by_feature

MODEL_DIR = Path(__file__).parent / "models"
model = joblib.load(MODEL_DIR / "attrition_model.joblib")
card = json.loads((MODEL_DIR / "model_card.json").read_text(encoding="utf-8"))


def risk_band(p):
    if p >= card["risk_bands"]["high"]:
        return "High"
    return "Low" if p < card["risk_bands"]["low"] else "Medium"


def predict_employee(values):
    """values: {feature: value} -> (probability of leaving, risk band, contributions Series)."""
    row = pd.DataFrame([values])[FEATURES]
    p = float(model.predict_proba(row)[0, 1])
    contrib, _ = shap_by_feature(model, row)
    return p, risk_band(p), contrib.iloc[0]


def explain_plot(contrib, values, top=8):
    c = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([f"{f} = {values[f]}" for f in c.index], c.values,
            color=["#DD5555" if v > 0 else "#4C72B0" for v in c.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Contribution (log-odds)   red: towards leaving   blue: towards staying")
    fig.tight_layout()
    return fig


def run(*args):
    values = dict(zip(FEATURES, args))
    p, band, contrib = predict_employee(values)
    return {"Leaves": p, "Stays": 1 - p}, f"**{band} risk** ({p:.0%} probability of leaving)", explain_plot(contrib, values)


def build_inputs():
    ranges, defaults = card["numeric_ranges"], card["categorical_defaults"]
    inputs = []
    for f in FEATURES:
        if f in NUMERIC:
            lo, hi, med = ranges[f]
            inputs.append(gr.Slider(lo, hi, value=med, step=1, label=f))
        else:
            choices = ORDINAL.get(f) or NOMINAL[f]
            inputs.append(gr.Dropdown(choices, value=defaults[f], label=f))
    return inputs


with gr.Blocks(title="Employee Attrition Risk") as demo:
    gr.Markdown(
        "# Employee attrition risk\n"
        f"**{card['model']}**, with test ROC-AUC {card['test_metrics']['ROC-AUC']:.3f}. It was trained on a "
        "[synthetic Kaggle HR dataset](https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset), "
        "so treat it as a demo, not a tool for real decisions about people."
    )
    with gr.Row():
        with gr.Column(scale=3):
            inputs = build_inputs()
            button = gr.Button("Predict", variant="primary")
        with gr.Column(scale=2):
            band = gr.Markdown()
            probs = gr.Label(label="Prediction")
            plot = gr.Plot(label="Why? Top drivers for this employee (SHAP)")
    button.click(run, inputs, [probs, band, plot])

if __name__ == "__main__":
    demo.launch()
