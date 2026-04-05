"""
Interactive briefing: Assessing AI System Vulnerabilities
Guided walkthrough for senior policymakers and decision-makers.
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuration & CSS
st.set_page_config(page_title="AI Vulnerability Briefing", layout="wide")

st.markdown("""
<style>
    /* Force deep dark background: Intelligence Dashboard Aesthetic */
    .stApp, header[data-testid="stHeader"] { background-color: #09090B; color: #F4F4F5; }
    
    /* Hide the Streamlit sidebar and its toggle button completely */
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
    
    /* Typography */
    * { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important; }
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 600 !important; letter-spacing: -0.02em; }
    p, span, label, div { color: #D4D4D8; }
    
    /* Prototype Banner */
    .prototype-banner {
        background-color: #27272A;
        color: #A1A1AA;
        text-align: center;
        padding: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border-bottom: 1px solid #3F3F46;
        margin-top: -60px;
        margin-bottom: 24px;
        margin-left: -4rem;
        margin-right: -4rem;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #18181B;
        border: 1px solid #27272A;
        border-radius: 4px;
        padding: 16px 24px;
    }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; }
    
    /* Executive Summary Box */
    .exec-summary {
        background-color: #18181B;
        border-left: 3px solid #3B82F6;
        padding: 20px 24px;
        margin-bottom: 32px;
        color: #A1A1AA;
        line-height: 1.6;
        border-top: 1px solid #27272A;
        border-right: 1px solid #27272A;
        border-bottom: 1px solid #27272A;
    }
    .exec-summary strong { color: #F4F4F5; font-weight: 600; }
    
    /* Custom Alert Boxes */
    .alert-critical { border-left: 3px solid #EF4444; background: rgba(239, 68, 68, 0.1); padding: 16px; color: #FCA5A5; margin-top: 16px; }
    .alert-warning { border-left: 3px solid #F59E0B; background: rgba(245, 158, 11, 0.1); padding: 16px; color: #FCD34D; margin-top: 16px; }
    .alert-stable { border-left: 3px solid #10B981; background: rgba(16, 185, 129, 0.1); padding: 16px; color: #6EE7B7; margin-top: 16px; }
    
    /* Hide Plotly Toolbar */
    .modebar { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ML & Math Logic
C0 = "#22D3EE"         # Cyan (Legitimate)
C1 = "#FB7185"         # Rose (Malicious/Fraudulent)
C0_BG = "rgba(34, 211, 238, 0.08)"
C1_BG = "rgba(251, 113, 133, 0.08)"

class Net(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2,h), nn.ReLU(), nn.Linear(h,h), nn.ReLU(), nn.Linear(h,2))
    def forward(self, x): 
        return self.net(x)

@st.cache_data(show_spinner=False)
def get_data(name, n, nz):
    if name == "Complex (Non-linear)": X, y = make_moons(n_samples=n, noise=nz, random_state=42)
    elif name == "Enclosed (Perimeter)": X, y = make_circles(n_samples=n, noise=nz, factor=0.5, random_state=42)
    else: X, y = make_classification(n_samples=n, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, flip_y=nz, random_state=42)
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.3, random_state=42)

@st.cache_resource(show_spinner=False)
def train_model(_X, _y, hidden_nodes):
    m = Net(hidden_nodes)
    o = torch.optim.Adam(m.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(_X, dtype=torch.float32), torch.tensor(_y, dtype=torch.long)
    for _ in range(250):
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
    
    if "Baseline" in attack_type:
        return X + np.random.uniform(-eps, eps, size=X.shape)
    elif "Single-step" in attack_type:
        nn.CrossEntropyLoss()(m(Xt), yt).backward()
        return (Xt + eps * Xt.grad.sign()).detach().numpy()
    elif "Multi-step" in attack_type:
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
    xs = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 150)
    ys = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 150)
    xx, yy = np.meshgrid(xs, ys)
    with torch.no_grad():
        Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)).argmax(1).numpy().reshape(xx.shape)
    
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=Z, showscale=False, colorscale=[[0, C0_BG], [1, C1_BG]],
        contours=dict(showlines=True, coloring="fill"),
        line=dict(width=1, color="rgba(255,255,255,0.1)"), hoverinfo="skip"))
    
    colors = [C0 if yi == 0 else C1 for yi in y]
    fig.add_trace(go.Scatter(
        x=X[:,0], y=X[:,1], mode="markers", hoverinfo="skip",
        marker=dict(color=colors, size=7, line=dict(width=0.8, color="#09090B"))))
    
    if arrows_from is not None:
        for i in range(len(X)):
            if np.linalg.norm(X[i] - arrows_from[i]) > 0.01:
                fig.add_annotation(
                    x=X[i,0], y=X[i,1], ax=arrows_from[i,0], ay=arrows_from[i,1],
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="rgba(255, 255, 255, 0.6)")

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#F4F4F5"), x=0.5),
        height=450, showlegend=False, margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

# Application Layout
st.markdown('<div class="prototype-banner">Prototype Application: For Policy Briefing Purposes Only</div>', unsafe_allow_html=True)

st.title("Assessing AI System Vulnerabilities")

st.markdown("""
<div class="exec-summary">
    <strong>Executive Briefing: Adversarial Brittleness</strong><br>
    Machine learning systems rely on mathematical boundaries to distinguish between legitimate activity (Cyan) and malicious activity (Rose). 
    While highly accurate in controlled environments, these systems possess a critical blind spot: an adversary can apply microscopic, 
    calculated modifications to input data, forcing the AI to make incorrect decisions with high confidence. 
</div>
""", unsafe_allow_html=True)

# Control Panel: Moved from sidebar to top columns
ctrl1, ctrl2, ctrl3 = st.columns(3)

with ctrl1:
    st.markdown("### 1. System Environment")
    ds_name = st.selectbox("Underlying Data Pattern", ["Complex (Non-linear)", "Enclosed (Perimeter)", "Standard (Linear)"])
    noise = st.slider("Data Ambiguity (Overlap)", 0.05, 0.40, 0.15, 0.05, help="How closely legitimate and malicious cases resemble each other in the real world.")

with ctrl2:
    st.markdown("### 2. Defense Capability")
    hidden_nodes = st.slider("AI Internal Complexity", 4, 64, 16, 4, help="The volume of computational resources the AI uses to draw its decision boundary.")

with ctrl3:
    st.markdown("### 3. Threat Scenario")
    attack_type = st.selectbox("Attack Methodology", [
        "Advanced Multi-step (Industry Threat Standard)", 
        "Basic Single-step (Fast Approximation)", 
        "Random Hardware Noise (Baseline)"
    ])
    eps = st.slider("Attacker's Modification Budget", 0.0, 1.5, 0.3, 0.05, help="The limit on how much an attacker can alter the input data before a human notices.")

st.markdown("<hr style='border-color: #27272A; margin-top: 10px; margin-bottom: 30px;'>", unsafe_allow_html=True)

# Execution Pipeline
X_train, X_test, y_train, y_test = get_data(ds_name, 400, noise)
model = train_model(X_train, y_train, hidden_nodes)

clean_acc = get_accuracy(model, X_test, y_test)
X_adv = generate_attack(model, X_test, y_test, attack_type, eps)
adv_acc = get_accuracy(model, X_adv, y_test)
acc_drop = clean_acc - adv_acc

# Telemetry
col1, col2, col3 = st.columns(3)
col1.metric("Baseline Reliability", f"{clean_acc:.1f}%")
col2.metric("Reliability Under Attack", f"{adv_acc:.1f}%")
col3.metric("System Degradation", f"{acc_drop:.1f} pts", delta=f"-{acc_drop:.1f}%", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# Visual Intelligence
c1, c2 = st.columns(2)

with c1:
    st.plotly_chart(render_plot(X_test, y_test, model, "Baseline Operation (Secure Data)"), use_container_width=True, config={'displayModeBar': False})

with c2:
    st.plotly_chart(render_plot(X_adv, y_test, model, "Active Threat Scenario", arrows_from=X_test), use_container_width=True, config={'displayModeBar': False})

# Strategic Assessment Logic
if acc_drop > 15:
    st.markdown(f"""
    <div class="alert-critical">
        <strong>CRITICAL VULNERABILITY DETECTED:</strong> The attack successfully bypassed system logic, degrading reliability by {acc_drop:.1f} percentage points. 
        Note the vector lines on the right panel: the adversary applied minimal alterations to the data points, yet successfully pushed them across the AI's fixed decision boundary. 
        In a live environment, this results in unauthorized approvals and false flag alerts.
    </div>
    """, unsafe_allow_html=True)
elif acc_drop > 5:
    st.markdown(f"""
    <div class="alert-warning">
        <strong>MODERATE DEGRADATION:</strong> System reliability decreased by {acc_drop:.1f} percentage points. 
        While the core architecture did not entirely collapse, an adversary has successfully manipulated a statistically significant subset of outcomes. 
        Consider this threshold a failure for mission-critical deployments.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="alert-stable">
        <strong>SYSTEM STABLE:</strong> At the current attack parameter, the model's decision boundaries remain structurally sound. 
        To stress-test further, increase the "Attacker's Modification Budget" via the control panel to identify the system's breaking point.
    </div>
    """, unsafe_allow_html=True)
