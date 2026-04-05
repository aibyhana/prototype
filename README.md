# Adversarial Decision Boundary Explorer

Interactive Streamlit app that lets you train a neural network on a toy dataset and watch adversarial attacks break its predictions in real time.

Built for **Stanford SAFE** — helping policymakers build intuition about adversarial brittleness in ML systems.

## Run locally

```bash
uv sync
uv run streamlit run app.py
```

Or with pip:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Hugging Face Spaces

1. Create a new Space with the **Docker** SDK
2. Push these files to the repo
3. Add a `Dockerfile`:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
```

## Stack

- **PyTorch** — model training and gradients
- **ART** — adversarial attacks (FGSM, PGD)
- **Plotly** — interactive decision boundary plots
- **scikit-learn** — toy datasets and preprocessing
- **Streamlit** — the UI
