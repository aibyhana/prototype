import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. GovTech / Enterprise Dark Mode Configuration
st.set_page_config(page_title="AI Policy Briefing: Robustness", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Base Enterprise Dark Theme */
    .stApp { 
        background-color: #0B1120; /* Deep Navy/Slate */
        color: #F8FAFC;
    }
    
    * { font-family: 'Inter', sans-serif !important; }
    
    /* Hide Streamlit Defaults */
    header[data-testid="stHeader"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }[data-testid="stSidebar"] { display: none !important; }
    .block-container { padding-top: 3rem !important; max-width: 1200px !important; }
    
    /* Typography */
    h1 { color: #FFFFFF !important; font-weight: 700 !important; font-size: 2.2rem !important; letter-spacing: -0.02em !important; margin-bottom: 0.5rem !important; }
    p { font-size: 1.05rem; line-height: 1.6; color: #94A3B8; }
    
    /* Executive Summary Card */
    .exec-summary {
        background-color: #1E293B; /* Slightly lighter slate */
        border-left: 4px solid #3B82F6; /* Trust Blue */
        border-radius: 4px 8px 8px 4px;
        padding: 24px 32px;
        margin-bottom: 32px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .exec-title {
        color: #E2E8F0;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
    }
    
    /* Control Panel */
    .control-panel {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }
    
    /* Metrics Override for Dark Mode */
    div[data-testid="stMetric"] { 
        background-color: #0F172A; 
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px; 
    }
    div[data-testid="stMetricLabel"] { color: #94A3B8 !important; font-weight: 600 !important; }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700 !important; font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# 2. ML Engine & Authoritative Colors
C_SAFE = "#3B82F6"       # Compliance Blue
C_THREAT = "#EF4444"     # Alert Red
C_SAFE_BG = "rgba(59, 130, 246, 0.08)"
C_THREAT_BG = "rgba(239, 68, 68, 0.08)"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x): return self.net(x)

@st.cache_data(show_spinner=False)
def get_data():
    X, y = make_moons(n_samples=300, noise=0.12, random_state=42)
    return train_test_split(StandardScaler().fit_transform(X), y, test_size=0.3, random_state=42)

@st.cache_resource(show_spinner=False)
def train_model(_X, _y):
    m = Net()
    o = torch.optim.Adam(m.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(_X, dtype=torch.float32), torch.tensor(_y, dtype=torch.long)
    for _ in range(300):
        o.zero_grad(); l = L(m(Xt), yt); l.backward(); o.step()
    m.eval()
    return m

def get_metrics(m, X, y):
    with torch.no_grad():
        preds = m(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()
        return (preds == y).mean() * 100, ((preds == 0) & (y == 1)).sum()

def generate_attack(m, X, y, eps):
    if eps == 0.0: return X
    Xt = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    yt = torch.tensor(y, dtype=torch.long)
    Xa = Xt.clone().detach()
    alpha = eps / 4
    for _ in range(10):
        Xa.requires_grad_(True)
        nn.CrossEntropyLoss()(m(Xa), yt).backward()
        with torch.no_grad():
            Xa = Xt + torch.clamp(Xa + alpha * Xa.grad.sign() - Xt, -eps, eps)
    return Xa.detach().numpy()

def render_chart(X, y, model, title, subtitle, arrows_from=None):
    fig = go.Figure()
    xs = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 120)
    ys = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 120)
    xx, yy = np.meshgrid(xs, ys)
    with torch.no_grad():
        Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)).argmax(1).numpy().reshape(xx.shape)
    
    # Subtle background coloring for Dark Mode
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=Z, showscale=False, colorscale=[[0, C_SAFE_BG],[1, C_THREAT_BG]],
        contours=dict(showlines=False, coloring="fill"), hoverinfo="skip"))
    
    # Clean markers
    colors =[C_SAFE if yi == 0 else C_THREAT for yi in y]
    fig.add_trace(go.Scatter(
        x=X[:,0], y=X[:,1], mode="markers", hoverinfo="skip",
        marker=dict(color=colors, size=8, line=dict(width=1, color="#0B1120"), opacity=1.0)))
    
    # Arrows indicating data manipulation
    if arrows_from is not None:
        for i in range(len(X)):
            if np.linalg.norm(X[i] - arrows_from[i]) > 0.01:
                fig.add_annotation(
                    x=X[i,0], y=X[i,1], ax=arrows_from[i,0], ay=arrows_from[i,1],
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#64748B", opacity=0.9)

    fig.update_layout(
        title=dict(text=f"<span style='font-size:18px; font-weight:600; color:#F8FAFC;'>{title}</span><br><span style='font-size:13px;color:#94A3B8; font-weight:400;'>{subtitle}</span>", x=0.02, y=0.95),
        height=420, showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

# 3. UI Layout - Policy Briefing Style

st.title("Interactive Briefing: Why AI Robustness Matters")

st.markdown("""
<div class='exec-summary'>
    <div class='exec-title'>Executive Summary</div>
    <p style='margin: 0; color: #CBD5E1;'>
        There is a critical gap between what AI systems actually do and what regulatory frameworks assume they do. 
        AI models do not possess human context; they draw rigid mathematical boundaries based on training data. 
        As demonstrated below, bad actors can exploit this. By making invisible numerical alterations to a dataset, 
        an attacker can bypass AI safety guardrails entirely. <strong>Regulating AI effectively requires understanding how easily these mathematical boundaries can be manipulated.</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Process Models
X_train, X_test, y_train, y_test = get_data()
model = train_model(X_train, y_train)

# Interactive Control Panel
st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
st.markdown("<h4 style='margin-top:0; color:#F8FAFC; font-weight:600; font-size:1.1rem;'>Parameter Control: Adversarial Pressure</h4>", unsafe_allow_html=True)

col_slider, col_desc = st.columns([1, 1.5], gap="large")
with col_slider:
    eps = st.slider("Data Manipulation Severity (Epsilon)", 0.0, 1.0, 0.0, 0.05)
with col_desc:
    if eps == 0.0:
        msg = "<b>Baseline State:</b> The AI is processing unaltered data. The system operates accurately and safety boundaries are intact."
    elif eps <= 0.35:
        msg = f"<b>Mild Alteration (ε={eps:.2f}):</b> Minor dataset tweaks. <i>Real-world impact:</i> An attacker changes a few imperceptible pixels in an image to bypass a content moderation filter."
    elif eps <= 0.7:
        msg = f"<b>Moderate Alteration (ε={eps:.2f}):</b> Structured interference. <i>Real-world impact:</i> Specific tape placed on a stop sign, causing an autonomous vehicle to classify it as a speed limit sign."
    else:
        msg = f"<b>Severe Alteration (ε={eps:.2f}):</b> Complete dataset distortion. <i>Real-world impact:</i> Adversarial noise applied to financial records to force automated loan approval."
    st.info(msg, icon="ℹ️")

st.markdown("</div>", unsafe_allow_html=True)

# Metrics
clean_acc, clean_bypassed = get_metrics(model, X_test, y_test)
X_adv = generate_attack(model, X_test, y_test, eps)
adv_acc, adv_bypassed = get_metrics(model, X_adv, y_test)

m1, m2, m3 = st.columns(3)
m1.metric("Compliance Accuracy (Baseline)", f"{clean_acc:.1f}%")
m2.metric("Accuracy Under Exploitation", f"{adv_acc:.1f}%", f"{adv_acc - clean_acc:.1f}%")
m3.metric("Safety Failures (False Negatives)", f"{adv_bypassed}", f"+{adv_bypassed - clean_bypassed} bypassed", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# Visualizations
c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown("<div style='border: 1px solid #334155; border-radius: 12px; background: #0F172A;'>", unsafe_allow_html=True)
    st.plotly_chart(render_chart(X_test, y_test, model, "1. Expected Boundary Alignment", "Model correctly separating safe (blue) and unsafe (red) data."), use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div style='border: 1px solid #334155; border-radius: 12px; background: #0F172A;'>", unsafe_allow_html=True)
    if eps == 0.0:
        st.plotly_chart(render_chart(X_adv, y_test, model, "2. Adversarial Manipulation", "Increase the slider above to simulate an attack."), use_container_width=True, config={'displayModeBar': False})
    else:
        st.plotly_chart(render_chart(X_adv, y_test, model, "2. Boundary Failure", "Gray lines show unsafe data mathematically forced into the safe zone."), use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)
