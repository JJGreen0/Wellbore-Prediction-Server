"""Simple Flask application serving ML predictions."""

from __future__ import annotations

import os
from functools import lru_cache

import joblib
import numpy as np
from flask import Flask, jsonify, request, render_template


app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "Models")


@lru_cache(maxsize=None)
def load_model(name: str):
    """Load a model by short name from the Models directory."""
    filename = f"{name}_50m_pipeline.pkl"
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@app.route("/")
def hello_world():
    return "Hello, Flask!"


@app.route("/models")
def list_models():
    """Return available model names."""
    names = []
    for filename in os.listdir(MODEL_DIR):
        if filename.endswith("_50m_pipeline.pkl"):
            names.append(filename.split("_", 1)[0])
    return jsonify({"models": sorted(names)})


@app.route("/demo")
def demo_page():
    """Serve the simple demo page."""
    return render_template("demo.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Return a single prediction.

    Expected JSON body::
        {
            "model": "BDTI",
            "features": [f1, f2, ...]
        }
    """

    data = request.get_json(silent=True) or {}
    model_name = data.get("model")
    features = data.get("features")

    if not model_name or features is None:
        return jsonify({"error": "model and features are required"}), 400

    model = load_model(model_name)
    if model is None:
        return jsonify({"error": f"model '{model_name}' not found"}), 404

    X = np.asarray([features])
    prediction = model.predict(X)[0]
    return jsonify({"prediction": float(prediction)})


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """Return batch predictions.

    Expected JSON body::
        {
            "model": "BDTI",
            "inputs": [[f1, f2, ...], [...]]
        }
    """

    data = request.get_json(silent=True) or {}
    model_name = data.get("model")
    inputs = data.get("inputs")

    if not model_name or inputs is None:
        return jsonify({"error": "model and inputs are required"}), 400

    model = load_model(model_name)
    if model is None:
        return jsonify({"error": f"model '{model_name}' not found"}), 404

    X = np.asarray(inputs)
    predictions = model.predict(X).tolist()
    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    # built‑in dev server
    app.run(debug=True, port=5000)
