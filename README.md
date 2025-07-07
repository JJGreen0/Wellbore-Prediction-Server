# Wellbore-Prediction-Server

Simple Flask server that exposes a REST API for making predictions using the
pre‑trained 50&nbsp;m models stored in the `Models` directory.

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the server

```bash
python app.py
```

By default the server listens on port `5000`.

## API

### `POST /predict`

Return a single prediction. The request body must contain the model name and a
list of feature values:

```json
{
  "model": "BDTI",
  "features": [1.0, 2.0, 3.0]
}
```

The response is:

```json
{ "prediction": 0.123 }
```

### `POST /batch_predict`

Return predictions for a batch of inputs. The body accepts an array of feature
vectors:

```json
{
  "model": "BDTI",
  "inputs": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
}
```

And the response:

```json
{ "predictions": [0.1, 0.2] }
```
