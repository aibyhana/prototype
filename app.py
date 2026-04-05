"""
Stanford SAFE · Adversarial Brittleness Module
An interactive briefing tool for decision-makers.
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

# ── page setup ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="SAFE · Adversarial Brittleness", layout="wide")

# ── custom CSS — institutional, warm, zero AI slop ─────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=DM+Sans:wght@400;500;600&display=swap');

/* reset streamlit defaults */
.stApp { background: #FAFAF7; }
header[data-testid="stHeader"] { background: transparent; }

/* typography */
h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif !important; color: #1a1a1a; }
p, li, label, .stMarkdown, .stText, span { font-family: 'DM Sans', sans-serif !important; color: #2c2c2c; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: #1B2838;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] .stMarkdown {
    color: #E8E6E1 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: #B0ADA6 !important;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* metric cards */
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E5E2DB;
    border-radius: 6px;
    padding: 16px 20px;
}
div[data-testid="stMetric"] label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: #888 !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-family: 'Source Serif 4', serif !important; font-size: 1.9rem; color: #1a1a1a !important; }

/* primary button */
.stButton > button[kind="primary"] {
    background: #8B2500;
    border: none;
    color: white;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    letter-spacing: 0.03em;
    border-radius: 4px;
}
.stButton > button[kind="primary"]:hover {
    background: #A63000;
}

/* expanders */
.streamlit-expanderHeader { font-family: 'Source Serif 4', serif !important; font-weight: 600; }

/* hide streamlit branding */
#MainMenu, footer { visibility: hidden; }

/* narrative callout */
.narrative-box {
    background: #F0EDE6;
    border-left: 3px solid #8B2500;
    padding: 20px 24px;
    border-radius: 0 6px 6px 0;
    margin: 12px 0 24px 0;
    font-family: 'DM Sans', sans-serif;
    color: #2c2c2c;
    line-height: 1.65;
    font-size: 0.95rem;
}

/* step badge */
.step-badge {
    display: inline-block;
    background: #1B2838;
    color: #E8E6E1;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 3px;
    margin-bottom: 8px;
}

/* divider */
.section-rule {
    border: none;
    border-top: 1px solid #DDD9D0;
    margin: 36px 0 28px 0;
}
</style>
""", unsafe_allow_html=True)

# ── model / attack code ────────────────────────────────────────────────────

class Classifier(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 2),
        )
    def forward(self, x):
        return self.net(x)


def make_data(name, n, noise_level):
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


def train_clf(model, X, y, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    history = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
        history.append(loss.item())
    return history


def fgsm(model, X, y, eps):
    Xt = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    yt = torch.tensor(y, dtype=torch.long)
    loss = nn.CrossEntropyLoss()(model(Xt), yt)
    loss.backward()
    return (Xt + eps * Xt.grad.sign()).detach().numpy()


def pgd(model, X, y, eps, alpha, steps):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    Xa = Xt.clone()
    for _ in range(steps):
        Xa.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(Xa), yt)
        loss.backward()
        with torch.no_grad():
            Xa = Xt + torch.clamp(Xa + alpha * Xa.grad.sign() - Xt, -eps, eps)
    return Xa.detach().numpy()


def add_noise(X, eps):
    return X + np.random.uniform(-eps, eps, size=X.shape)


def predict_grid(model, X, res=180):
    pad = 1.0
    xs = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, res)
    ys = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, res)
    xx, yy = np.meshgrid(xs, ys)
    with torch.no_grad():
        Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32))
        Z = Z.argmax(1).numpy().reshape(xx.shape)
    return xx, yy, Z


def acc(model, X, y):
    with torch.no_grad():
        p = model(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()
    return (p == y).mean() * 100


# ── colours ─────────────────────────────────────────────────────────────────
CLR_A = "#2D6A4F"
CLR_B = "#9B2226"
CLR_A_BG = "rgba(45,106,79,0.12)"
CLR_B_BG = "rgba(155,34,38,0.12)"

# ── header ──────────────────────────────────────────────────────────────────

st.markdown("""
<div style="font-family:'DM Sans',sans-serif; font-size:0.78rem; font-weight:600;
            letter-spacing:0.1em; text-transform:uppercase; color:#8B2500; margin-bottom:4px;">
    Stanford SAFE · Interactive Module
</div>
""", unsafe_allow_html=True)

st.markdown("## How a tiny, invisible change can break an AI system")

st.markdown("""
<div class="narrative-box">
You're about to train a small AI classifier — the same kind of pattern-recognition
system that underpins medical diagnostics, loan approvals, and content moderation.
It will learn to separate two groups of data points. Then you'll apply an
<strong>adversarial attack</strong>: a mathematically crafted nudge, invisible to
the human eye, that causes the system to fail. The model doesn't change. The data
barely changes. But the predictions collapse.
</div>
""", unsafe_allow_html=True)

# ── sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Configuration")
    st.caption("Adjust these to explore different scenarios.")

    st.markdown('<div class="step-badge">Step 1 — Data</div>', unsafe_allow_html=True)
    dataset_name = st.selectbox("Pattern shape", ["Two Moons", "Concentric Circles", "Gaussian Blobs"], label_visibility="collapsed")
    n_samples = st.slider("Sample count", 200, 1000, 400, 100)
    noise_val = st.slider("Messiness", 0.05, 0.50, 0.20, 0.05)

    st.markdown('<div class="step-badge">Step 2 — Model</div>', unsafe_allow_html=True)
    hidden = st.slider("Complexity (hidden neurons)", 4, 64, 16, 4)
    n_epochs = st.slider("Training rounds", 50, 500, 200, 50)

    st.markdown('<div class="step-badge">Step 3 — Attack</div>', unsafe_allow_html=True)
    attack_name = st.selectbox("Attack method", ["FGSM (one-shot)", "PGD (iterative)", "Random noise (baseline)"], label_visibility="collapsed")
    eps = st.slider("Perturbation strength (ε)", 0.0, 2.0, 0.30, 0.05)

    pgd_steps, pgd_alpha = 10, 0.05
    if "PGD" in attack_name:
        pgd_steps = st.slider("Iteration steps", 1, 40, 10)

    st.markdown("---")
    go_btn = st.button("Run the experiment", type="primary", use_container_width=True)

# ── landing state ───────────────────────────────────────────────────────────

if not go_btn:
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("### Before you begin")
        st.markdown("""
Use the panel on the left to configure the experiment, then press
**Run the experiment**.

You'll see two plots side by side — one showing the AI's predictions
on clean data, and one showing what happens after the attack. The
coloured regions are the AI's "decision zones." Points that land in
the wrong zone get misclassified.

Try increasing the perturbation strength (ε) and watch the accuracy
number drop. That gap between clean and attacked accuracy is the
**brittleness** that current AI regulation needs to address.
        """)
    with col_r:
        st.markdown("### Key terms")
        st.markdown("""
**Decision boundary** — the line the AI draws to separate groups.
Think of it as the rule the system learned.

**Adversarial attack** — a deliberate, tiny modification to inputs
designed to fool the AI. Not random noise — mathematically optimised.

**FGSM** — Fast Gradient Sign Method. A single-step attack that
follows the steepest direction of error.

**PGD** — Projected Gradient Descent. A stronger, multi-step version
of FGSM. The standard stress-test in the research community.

**ε (epsilon)** — how much the attacker is allowed to change each
data point. Smaller ε = subtler attack.
        """)

    st.stop()

# ── experiment ──────────────────────────────────────────────────────────────

X, y, scaler = make_data(dataset_name, n_samples, noise_val)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

model = Classifier(hidden)
with st.spinner("Training the classifier…"):
    history = train_clf(model, X_tr, y_tr, n_epochs, 0.01)
model.eval()

clean_score = acc(model, X_te, y_te)

with st.spinner("Applying the attack…"):
    if "FGSM" in attack_name:
        X_adv = fgsm(model, X_te, y_te, eps)
        atk_label = "FGSM"
    elif "PGD" in attack_name:
        X_adv = pgd(model, X_te, y_te, eps, pgd_alpha, pgd_steps)
        atk_label = "PGD"
    else:
        X_adv = add_noise(X_te, eps)
        atk_label = "Random noise"

adv_score = acc(model, X_adv, y_te)
drop = clean_score - adv_score

# ── results ─────────────────────────────────────────────────────────────────

st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
st.markdown('<div class="step-badge">Results</div>', unsafe_allow_html=True)
st.markdown("### What the attack did to the model")

m1, m2, m3 = st.columns(3)
m1.metric("Accuracy on clean data", f"{clean_score:.1f}%")
m2.metric("Accuracy after attack", f"{adv_score:.1f}%")
m3.metric("Performance lost", f"{drop:.1f} points", delta=f"−{drop:.1f}pp", delta_color="inverse")

# ── plots ───────────────────────────────────────────────────────────────────

xx, yy, Z = predict_grid(model, X)
clr_pts = [CLR_A if yi == 0 else CLR_B for yi in y_te]

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Before attack — clean data", f"After {atk_label} attack — ε = {eps}"),
    horizontal_spacing=0.07,
)

for c in [1, 2]:
    fig.add_trace(go.Contour(
        x=np.linspace(xx.min(), xx.max(), Z.shape[1]),
        y=np.linspace(yy.min(), yy.max(), Z.shape[0]),
        z=Z, showscale=False,
        colorscale=[[0, CLR_A_BG], [1, CLR_B_BG]],
        contours=dict(showlines=True, coloring="fill"),
        line=dict(width=1.5, color="rgba(0,0,0,0.18)"),
        hoverinfo="skip",
    ), row=1, col=c)

fig.add_trace(go.Scatter(
    x=X_te[:, 0], y=X_te[:, 1], mode="markers",
    marker=dict(color=clr_pts, size=5.5, line=dict(width=0.4, color="white")),
    hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>",
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=X_adv[:, 0], y=X_adv[:, 1], mode="markers",
    marker=dict(color=clr_pts, size=5.5, line=dict(width=0.4, color="white")),
    hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>",
), row=1, col=2)

rng = np.random.RandomState(0)
idx = rng.choice(len(X_te), size=min(35, len(X_te)), replace=False)
for i in idx:
    fig.add_annotation(
        x=X_adv[i, 0], y=X_adv[i, 1],
        ax=X_te[i, 0], ay=X_te[i, 1],
        xref="x2", yref="y2", axref="x2", ayref="y2",
        showarrow=True, arrowhead=3, arrowsize=0.8, arrowwidth=0.7,
        arrowcolor="rgba(0,0,0,0.20)",
    )

fig.update_layout(
    height=480, showlegend=False,
    margin=dict(t=36, b=16, l=16, r=16),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", size=12),
)
fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
fig.update_annotations(font_size=13)

st.plotly_chart(fig, use_container_width=True)

# ── interpretation ──────────────────────────────────────────────────────────

st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
st.markdown('<div class="step-badge">Interpretation</div>', unsafe_allow_html=True)
st.markdown("### Why this matters for policy")

if drop > 15:
    verdict = (
        f"The attack reduced accuracy by <strong>{drop:.0f} percentage points</strong> — "
        f"a catastrophic failure. In a real system this could mean a medical "
        f"scan reading flipping from 'benign' to 'malignant', or a loan "
        f"approval reversing, with no visible change to the input."
    )
elif drop > 5:
    verdict = (
        f"Accuracy dropped by <strong>{drop:.0f} points</strong>. That's significant — "
        f"enough to cause material harm in high-stakes applications where "
        f"even a few percentage points of error translate to real consequences."
    )
elif drop > 0.5:
    verdict = (
        f"The drop was modest (<strong>{drop:.1f} points</strong>), but notice the attack "
        f"still moved points across the boundary. In safety-critical systems, "
        f"even a small targeted failure can be exploited."
    )
else:
    verdict = (
        "At this perturbation level the model held up. Try increasing ε "
        "or switching to PGD to see when it starts breaking."
    )

st.markdown(f"""
<div class="narrative-box">
{verdict}
<br><br>
The thin grey line in the plots is the <strong>decision boundary</strong> — the
rule the AI learned. Notice how close many data points sit to that line. An
adversarial attack doesn't need to move points far; it just needs to push them
across. Current AI systems offer no built-in guarantee about how close the
boundary is to the data, and most deployed models have never been tested against
these attacks.
<br><br>
This is the gap that regulation needs to close: mandatory adversarial robustness
testing before deployment in high-stakes domains.
</div>
""", unsafe_allow_html=True)

# ── training loss (tucked away) ─────────────────────────────────────────────

with st.expander("Technical detail — training loss curve"):
    lfig = go.Figure()
    lfig.add_trace(go.Scatter(
        y=history, mode="lines",
        line=dict(color="#1B2838", width=1.8),
    ))
    lfig.update_layout(
        xaxis_title="Training round", yaxis_title="Loss",
        height=260, margin=dict(t=8, b=36, l=44, r=16),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", size=11),
    )
    lfig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    lfig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    st.plotly_chart(lfig, use_container_width=True)

# ── footer ──────────────────────────────────────────────────────────────────

st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding: 8px 0 20px 0;">
    <span style="font-family:'DM Sans',sans-serif; font-size:0.78rem; color:#999; letter-spacing:0.04em;">
        Stanford SAFE · Prototype · Not for redistribution
    </span>
</div>
""", unsafe_allow_html=True)
