"""
Interactive briefing: Can AI systems be tricked?
Guided walkthrough for senior decision-makers.
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Configuration & CSS ──────────────────────────────────────────────────────

st.set_page_config(page_title="AI Vulnerability Briefing", layout="wide", initial_sidebar_state="expanded")

# Modern, sleek Dark Mode (Zinc/Slate palette)
st.markdown("""
<style>
    /* Force deep dark background */
    .stApp, header[data-testid="stHeader"] { background-color: #09090B; color: #F4F4F5; }
    
    /* Typography */
    * { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important; }
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 600 !important; letter-spacing: -0.02em; }
    p, span, label, div { color: #D4D4D8; }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #18181B;
        border: 1px solid #27272A;
        border-radius: 8px;
        padding: 16px 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; }
    
    /* Executive Summary Box */
    .exec-summary {
        background-color: #18181B;
        border-left: 4px solid #3B82F6;
        padding: 20px 24px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 24px;
        color: #A1A1AA;
        line-height: 1.6;
        border-top: 1px solid #27272A;
        border-right: 1px solid #27272A;
        border-bottom: 1px solid #27272A;
    }
    .exec-summary strong { color: #F4F4F5; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #18181B; border-right: 1px solid #27272A; }
    
    /* Sliders and Selectboxes overrides for dark mode blending */
    .stSlider > div > div > div { color: #F4F4F5 !important; }
</style>
""", unsafe_allow_html=True)

# ── ML & Math Logic ──────────────────────────────────────────────────────────

# Elegant Dark Mode Palette
C0 = "#22D3EE"         # Cyan (Legitimate)
C1 = "#FB7185"         # Rose (Fraudulent)
C0_BG = "rgba(34, 211, 238, 0.08)" # Faint Cyan for decision zone
C1_BG = "rgba(251, 113, 133, 0.08)" # Faint Rose for decision zone

class Net(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2,h), nn.ReLU(), nn.Linear(h,h), nn.ReLU(), nn.Linear(h,2))
    def forward(self, x): 
        return self.net(x)

@st.cache_data(show_spinner=False)
def get_data(name, n, nz):
    if name == "Moons (Non-linear)": X, y = make_moons(n_samples=n, noise=nz, random_state=42)
    elif name == "Circles (Enclosed)": X, y = make_circles(n_samples=n, noise=nz, factor=0.5, random_state=42)
    else: X, y = make_classification(n_samples=n, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, flip_y=nz, random_state=42)
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.3, random_state=42)

@st.cache_resource(show_spinner=False)
def train_model(_X, _y, hidden_nodes):
    m = Net(hidden_nodes)
    o = torch.optim.Adam(m.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(_X, dtype=torch.float32), torch.tensor(_y, dtype=torch.long)
    for _ in range(250): # Fast training loop
        o.zero_grad(); l = L(m(Xt), yt); l.backward(); o.step()
    m.eval()
    return m

def get_accuracy(m, X, y):
    with torch.no_grad():
        preds = m(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()
        return (preds == y).mean() * 100

def generate_attack(m, X, y, attack_type, eps):
    if eps == 0.0: return X
    
    Xt = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    yt = torch.tensor(y, dtype=torch.long)
    
    if "Noise" in attack_type:
        return X + np.random.uniform(-eps, eps, size=X.shape)
        
    elif "FGSM" in attack_type:
        nn.CrossEntropyLoss()(m(Xt), yt).backward()
        return (Xt + eps * Xt.grad.sign()).detach().numpy()
        
    elif "PGD" in attack_type:
        Xa = Xt.clone().detach()
        alpha = eps / 4
        for _ in range(10):
            Xa.requires_grad_(True)
            nn.CrossEntropyLoss()(m(Xa), yt).backward()
            with torch.no_grad():
                Xa = Xt + torch.clamp(Xa + alpha * Xa.grad.sign() - Xt, -eps, eps)
        return Xa.detach().numpy()

def render_plot(X, y, model, title, arrows_from=None):
    fig = go.Figure()
    
    # Draw Decision Boundary
    xs = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 150)
    ys = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 150)
    xx, yy = np.meshgrid(xs, ys)
    with torch.no_grad():
        Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)).argmax(1).numpy().reshape(xx.shape)
    
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=Z, showscale=False, colorscale=[[0, C0_BG], [1, C1_BG]],
        contours=dict(showlines=True, coloring="fill"),
        line=dict(width=1, color="rgba(255,255,255,0.1)"), hoverinfo="skip"))
    
    # Draw Data Points
    colors = [C0 if yi == 0 else C1 for yi in y]
    fig.add_trace(go.Scatter(
        x=X[:,0], y=X[:,1], mode="markers",
        marker=dict(color=colors, size=7, line=dict(width=0.8, color="#09090B")),
        hovertemplate="Feature 1: %{x:.2f}<br>Feature 2: %{y:.2f}<extra></extra>"))
    
    # Draw Attack Arrows (Slightly opaque white in dark mode)
    if arrows_from is not None:
        for i in range(len(X)):
            if np.linalg.norm(X[i] - arrows_from[i]) > 0.01: # Only draw if point actually moved
                fig.add_annotation(
                    x=X[i,0], y=X[i,1], ax=arrows_from[i,0], ay=arrows_from[i,1],
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="rgba(255, 255, 255, 0.4)")

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#F4F4F5"), x=0.5),
        height=450, showlegend=False, margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

# ── Sidebar UI (Controls) ────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 1. The Environment")
    ds_name = st.selectbox("Data Pattern", ["Moons (Non-linear)", "Circles (Enclosed)", "Clusters (Linear)"])
    noise = st.slider("Data Complexity (Overlap)", 0.05, 0.40, 0.15, 0.05)
    
    st.markdown("### 2. The AI Model")
    hidden_nodes = st.slider("Model Capacity", 4, 64, 16, 4)
    
    st.markdown("### 3. The Threat")
    attack_type = st.selectbox("Attack Algorithm", ["FGSM (Single-step)", "PGD (Multi-step / SOTA)", "Random Noise (Baseline)"])
    eps = st.slider("Attack Strength (Epsilon)", 0.0, 1.5, 0.3, 0.05)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("💡 **Tip:** Adjust sliders to see the model and attacks update in real-time.")

# ── Main Dashboard ───────────────────────────────────────────────────────────

st.title("Can AI Systems Be Tricked?")

st.markdown("""
<div class="exec-summary">
    <strong>Executive Briefing: Adversarial Brittleness</strong><br>
    Machine learning models draw mathematical boundaries to separate safe data (Cyan) from fraudulent/malicious data (Rose). 
    While highly accurate on normal data, these models possess a blind spot: an attacker can apply tiny, mathematically optimized 
    perturbations to inputs, forcing the AI to confidently make the wrong decision. 
</div>
""", unsafe_allow_html=True)

# Process Data & Model
X_train, X_test, y_train, y_test = get_data(ds_name, 400, noise)
model = train_model(X_train, y_train, hidden_nodes)

# Process Attacks
clean_acc = get_accuracy(model, X_test, y_test)
X_adv = generate_attack(model, X_test, y_test, attack_type, eps)
adv_acc = get_accuracy(model, X_adv, y_test)
acc_drop = clean_acc - adv_acc

# Metrics Row
col1, col2, col3 = st.columns(3)
col1.metric("Original AI Accuracy", f"{clean_acc:.1f}%", help="Accuracy on untouched data.")
col2.metric("Accuracy Under Attack", f"{adv_acc:.1f}%", help="Accuracy on manipulated data.")
col3.metric("System Degradation", f"{acc_drop:.1f} pts", delta=f"-{acc_drop:.1f}%", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# Visualizations
c1, c2 = st.columns(2)

with c1:
    st.plotly_chart(render_plot(X_test, y_test, model, "Normal Operation (Clean Data)"), use_container_width=True)

with c2:
    st.plotly_chart(render_plot(X_adv, y_test, model, f"Under Attack ({attack_type})", arrows_from=X_test), use_container_width=True)

# Dynamic plain-english conclusion
if acc_drop > 15:
    st.error(f"**Critical Vulnerability:** The attack bypassed the system, dropping accuracy by {acc_drop:.1f}%. Notice the arrows on the right graph—the attacker barely moved the data points, but successfully pushed them across the AI's decision boundary.")
elif acc_drop > 5:
    st.warning(f"**Moderate Degradation:** The AI lost {acc_drop:.1f}% accuracy. While the system didn't entirely collapse, an attacker successfully compromised a meaningful subset of decisions.")
else:
    st.success("**System Resilient:** At this current attack strength, the model's decision boundaries held. Try increasing the Attack Strength in the sidebar.")
