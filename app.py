"""
Stanford SAFE — Adversarial Decision Boundary Explorer
Interactive module for introducing policymakers to adversarial ML concepts.
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Adversarial Decision Boundary Explorer",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Adversarial Decision Boundary Explorer")
st.markdown(
    "Train a small neural network on a toy dataset, then watch how "
    "adversarial attacks shift the decision boundary and fool the model."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("1 · Dataset")
dataset_name = st.sidebar.selectbox(
    "Choose a dataset",
    ["Two Moons", "Concentric Circles", "Gaussian Blobs"],
)
n_samples = st.sidebar.slider("Number of samples", 200, 1000, 400, step=100)
noise = st.sidebar.slider("Noise level", 0.05, 0.50, 0.20, step=0.05)

st.sidebar.header("2 · Model")
hidden_size = st.sidebar.slider("Hidden neurons", 4, 64, 16, step=4)
epochs = st.sidebar.slider("Training epochs", 50, 500, 200, step=50)
lr = st.sidebar.select_slider(
    "Learning rate",
    options=[0.001, 0.005, 0.01, 0.05, 0.1],
    value=0.01,
)

st.sidebar.header("3 · Attack")
attack_type = st.sidebar.selectbox("Attack type", ["FGSM", "PGD", "Random Noise"])
epsilon = st.sidebar.slider("Perturbation size (ε)", 0.0, 2.0, 0.3, step=0.05)
if attack_type == "PGD":
    pgd_steps = st.sidebar.slider("PGD steps", 1, 40, 10)
    pgd_alpha = st.sidebar.slider("PGD step size (α)", 0.01, 0.5, 0.05, step=0.01)

run = st.sidebar.button("▶  Train & Attack", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_data(name, n, noise_level):
    if name == "Two Moons":
        X, y = make_moons(n_samples=n, noise=noise_level, random_state=42)
    elif name == "Concentric Circles":
        X, y = make_circles(n_samples=n, noise=noise_level, factor=0.5, random_state=42)
    else:
        X, y = make_classification(
            n_samples=n, n_features=2, n_redundant=0,
            n_informative=2, n_clusters_per_class=1,
            flip_y=noise_level, random_state=42,
        )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler


class SimpleNet(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_model(model, X, y, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        out = model(X_t)
        loss = loss_fn(out, y_t)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def fgsm_attack(model, X, y, eps):
    X_t = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    y_t = torch.tensor(y, dtype=torch.long)
    loss_fn = nn.CrossEntropyLoss()
    out = model(X_t)
    loss = loss_fn(out, y_t)
    loss.backward()
    perturbation = eps * X_t.grad.sign()
    X_adv = X_t + perturbation
    return X_adv.detach().numpy()


def pgd_attack(model, X, y, eps, alpha, steps):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    X_adv = X_t.clone().detach()
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        X_adv.requires_grad_(True)
        out = model(X_adv)
        loss = loss_fn(out, y_t)
        loss.backward()
        with torch.no_grad():
            X_adv = X_adv + alpha * X_adv.grad.sign()
            # project back into eps-ball
            delta = torch.clamp(X_adv - X_t, min=-eps, max=eps)
            X_adv = X_t + delta
    return X_adv.detach().numpy()


def random_noise_attack(X, eps):
    noise = np.random.uniform(-eps, eps, size=X.shape)
    return X + noise


def get_decision_boundary(model, scaler, X, resolution=200):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_t = torch.tensor(grid, dtype=torch.float32)
    with torch.no_grad():
        preds = model(grid_t).argmax(dim=1).numpy()
    return xx, yy, preds.reshape(xx.shape)


def accuracy(model, X, y):
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_t).argmax(dim=1).numpy()
    return (preds == y).mean() * 100


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if run:
    # generate data
    X, y, scaler = generate_data(dataset_name, n_samples, noise)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train
    model = SimpleNet(hidden_size)
    with st.spinner("Training model..."):
        losses = train_model(model, X_train, y_train, epochs, lr)
    model.eval()

    clean_acc = accuracy(model, X_test, y_test)

    # attack
    with st.spinner("Running attack..."):
        if attack_type == "FGSM":
            X_test_adv = fgsm_attack(model, X_test, y_test, epsilon)
        elif attack_type == "PGD":
            X_test_adv = pgd_attack(model, X_test, y_test, epsilon, pgd_alpha, pgd_steps)
        else:
            X_test_adv = random_noise_attack(X_test, epsilon)

    adv_acc = accuracy(model, X_test_adv, y_test)

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Clean accuracy", f"{clean_acc:.1f}%")
    col2.metric("Post-attack accuracy", f"{adv_acc:.1f}%")
    col3.metric(
        "Accuracy drop",
        f"{clean_acc - adv_acc:.1f}pp",
        delta=f"-{clean_acc - adv_acc:.1f}pp",
        delta_color="inverse",
    )

    # -----------------------------------------------------------------------
    # Decision boundary plots
    # -----------------------------------------------------------------------
    xx, yy, Z = get_decision_boundary(model, scaler, X)

    colors_clean = ["#3b82f6" if yi == 0 else "#ef4444" for yi in y_test]
    colors_adv = ["#3b82f6" if yi == 0 else "#ef4444" for yi in y_test]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Clean test data", f"After {attack_type} attack (ε={epsilon})"),
        horizontal_spacing=0.08,
    )

    # decision regions (same model, so same boundary on both)
    for col_idx in [1, 2]:
        fig.add_trace(
            go.Contour(
                x=np.linspace(xx.min(), xx.max(), Z.shape[1]),
                y=np.linspace(yy.min(), yy.max(), Z.shape[0]),
                z=Z,
                showscale=False,
                colorscale=[[0, "rgba(59,130,246,0.15)"], [1, "rgba(239,68,68,0.15)"]],
                contours=dict(showlines=True, coloring="fill"),
                line=dict(width=1, color="rgba(0,0,0,0.3)"),
                hoverinfo="skip",
            ),
            row=1, col=col_idx,
        )

    # clean points
    fig.add_trace(
        go.Scatter(
            x=X_test[:, 0], y=X_test[:, 1],
            mode="markers",
            marker=dict(color=colors_clean, size=6, line=dict(width=0.5, color="white")),
            name="Clean",
            hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}",
        ),
        row=1, col=1,
    )

    # adversarial points
    fig.add_trace(
        go.Scatter(
            x=X_test_adv[:, 0], y=X_test_adv[:, 1],
            mode="markers",
            marker=dict(color=colors_adv, size=6, line=dict(width=0.5, color="white")),
            name="Adversarial",
            hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}",
        ),
        row=1, col=2,
    )

    # perturbation arrows (subset for clarity)
    arrow_indices = np.random.choice(len(X_test), size=min(40, len(X_test)), replace=False)
    for i in arrow_indices:
        fig.add_annotation(
            x=X_test_adv[i, 0], y=X_test_adv[i, 1],
            ax=X_test[i, 0], ay=X_test[i, 1],
            xref="x2", yref="y2", axref="x2", ayref="y2",
            showarrow=True,
            arrowhead=2, arrowsize=1, arrowwidth=0.8,
            arrowcolor="rgba(0,0,0,0.25)",
        )

    fig.update_layout(
        height=500,
        showlegend=False,
        margin=dict(t=40, b=20, l=20, r=20),
        template="plotly_white",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------------
    # Training loss curve
    # -----------------------------------------------------------------------
    with st.expander("Training loss curve"):
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(
            y=losses, mode="lines",
            line=dict(color="#6366f1", width=2),
        ))
        loss_fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Cross-entropy loss",
            height=300,
            margin=dict(t=10, b=40, l=40, r=20),
            template="plotly_white",
        )
        st.plotly_chart(loss_fig, use_container_width=True)

    # -----------------------------------------------------------------------
    # Explanation
    # -----------------------------------------------------------------------
    with st.expander("What just happened?"):
        st.markdown(f"""
**You trained** a small neural network with {hidden_size} hidden neurons
on {n_samples} data points from the *{dataset_name}* dataset.

**The model learned** a decision boundary — the line separating the blue
class from the red class — and achieved **{clean_acc:.1f}%** accuracy on
held-out test data.

**Then we attacked it** using **{attack_type}** with ε = {epsilon}. This
shifted each test point by a tiny amount — often imperceptible — but enough
to push many points across the decision boundary.

**The result:** accuracy dropped to **{adv_acc:.1f}%**. The model didn't
change. The data barely changed. But the predictions collapsed.

This is what *adversarial brittleness* looks like. The model's confidence
masks a fundamental fragility near its decision boundaries.
        """)

else:
    st.info("👈 Configure the settings in the sidebar and press **Train & Attack**.")
    st.markdown(
        "This tool lets you see — hands-on — how a simple adversarial "
        "perturbation can completely break a machine learning model's predictions, "
        "even when the changes to the input data are almost invisible."
    )
