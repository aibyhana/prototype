import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Configuration & Premium Textured CSS
st.set_page_config(page_title="AI Threat Assessment", layout="wide")

st.markdown("""
<style>
    /* Premium Dark Blue Textured Background */
    .stApp { 
        background-color: #060B14; 
        background-image: radial-gradient(rgba(148, 163, 184, 0.05) 1px, transparent 1px);
        background-size: 24px 24px;
        color: #E2E8F0; 
    }
    header[data-testid="stHeader"] { background-color: transparent !important; }
    
    /* Remove sidebar entirely */
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
    
    /* Typography */
    * { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; }
    h1 { color: #FFFFFF !important; font-weight: 700 !important; font-size: 2.2rem !important; margin-bottom: 0.2rem !important; }
    h3 { color: #94A3B8 !important; font-weight: 400 !important; font-size: 1.1rem !important; margin-bottom: 2rem !important; }
    p { font-size: 1.05rem; line-height: 1.6; color: #CBD5E1; }
    
    /* Frosted Glass Boxes */
    .glass-box {
        background-color: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 24px;
    }
    
    /* BLUF Box Specifics */
    .bluf-box {
        border-left: 4px solid #F43F5E;
        margin-bottom: 32px;
    }
    .bluf-text { color: #F8FAFC; font-weight: 600; font-size: 1.1rem; letter-spacing: 0.05em; margin-bottom: 8px; }
    
    /* Dynamic Threat Context Box */
    .threat-context {
        background-color: rgba(56, 189, 248, 0.05);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 6px;
        padding: 16px;
        margin-top: -10px;
        margin-bottom: 32px;
        color: #BAE6FD;
        font-size: 0.95rem;
    }
    
    /* Metric styling */
    div[data-testid="stMetric"] { 
        background-color: rgba(15, 23, 42, 0.6); 
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 16px 20px; 
    }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700 !important; }
    
    /* Custom Slider */
    .stSlider > div > div > div { color: #FFFFFF !important; font-weight: 600; font-size: 1.1rem; }
    
    /* Hide Plotly Toolbar */
    .modebar { display: none !important; }
</style>
""", unsafe_allow_html=True)

# 2. Mathematical Engine & AI Setup
C_SAFE = "#38BDF8"       # Light Blue
C_THREAT = "#F43F5E"     # Rose Red
C_SAFE_BG = "rgba(56, 189, 248, 0.05)"
C_THREAT_BG = "rgba(244, 63, 94, 0.05)"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x): return self.net(x)

@st.cache_data(show_spinner=False)
def get_data():
    X, y = make_moons(n_samples=300, noise=0.12, random_state=42)
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.3, random_state=42)

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
        acc = (preds == y).mean() * 100
        threats_bypassed = ((preds == 0) & (y == 1)).sum()
        return acc, threats_bypassed

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

def render_plot(X, y, model, title, subtitle, arrows_from=None):
    fig = go.Figure()
    
    xs = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 120)
    ys = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 120)
    xx, yy = np.meshgrid(xs, ys)
    with torch.no_grad():
        Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)).argmax(1).numpy().reshape(xx.shape)
    
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=Z, showscale=False, colorscale=[[0, C_SAFE_BG], [1, C_THREAT_BG]],
        contours=dict(showlines=True, coloring="fill"),
        line=dict(width=1, color="rgba(255,255,255,0.1)"), hoverinfo="skip"))
    
    colors = [C_SAFE if yi == 0 else C_THREAT for yi in y]
    fig.add_trace(go.Scatter(
        x=X[:,0], y=X[:,1], mode="markers", hoverinfo="skip",
        marker=dict(color=colors, size=8, line=dict(width=1, color="#060B14"))))
    
    if arrows_from is not None:
        for i in range(len(X)):
            if np.linalg.norm(X[i] - arrows_from[i]) > 0.01:
                fig.add_annotation(
                    x=X[i,0], y=X[i,1], ax=arrows_from[i,0], ay=arrows_from[i,1],
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="rgba(255, 255, 255, 0.9)")

    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:12px;color:#94A3B8'>{subtitle}</span>", x=0.05, y=0.95),
        height=480, showlegend=False, margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

# 3. Layout and Briefing Narrative

st.title("Threat Assessment: AI Decision Boundaries (Simple Prototype)")
st.markdown("### Policy Briefing on Machine Learning Vulnerabilities")

st.markdown("""
<div class="glass-box bluf-box">
    <div class="bluf-text">BOTTOM LINE UP FRONT</div>
    <p style="margin: 0;">
        Machine learning models do not actually understand the data they process. They simply draw a mathematical line between safe and dangerous profiles. Adversaries exploit this limitation. By making tiny numerical tweaks to malicious data, an attacker can push their profile across that line. The AI will then approve the threat with absolute confidence.
    </p>
</div>
""", unsafe_allow_html=True)

# Process Data Live
X_train, X_test, y_train, y_test = get_data()
model = train_model(X_train, y_train)

# Interactive Component
st.markdown("<h4 style='color: #FFFFFF; margin-bottom: 10px;'>ESCALATE THREAT LEVEL</h4>", unsafe_allow_html=True)
eps = st.slider("Intensity of Adversarial Evasion Tactics", 0.0, 1.0, 0.0, 0.05, label_visibility="collapsed")

# Dynamic Real-World Examples
if eps == 0.0:
    example_text = "Level 0: No manipulation. The AI is processing unaltered, standard data."
elif eps <= 0.35:
    example_text = f"Level {eps:.2f}: Minor digital manipulation. Real-world equivalent is changing a few pixels in a photo or subtly altering the background static in an audio recording."
elif eps <= 0.7:
    example_text = f"Level {eps:.2f}: Moderate physical manipulation. Real-world equivalent is placing a small piece of dark tape on a stop sign to confuse a self-driving car."
else:
    example_text = f"Level {eps:.2f}: Heavy structural manipulation. Real-world equivalent is applying a targeted digital noise filter over a medical X-ray to force a misdiagnosis."

st.markdown(f"""
<div class="threat-context">
    <strong>Real-World Example:</strong> {example_text}
</div>
""", unsafe_allow_html=True)

# Computations
clean_acc, clean_bypassed = get_metrics(model, X_test, y_test)
X_adv = generate_attack(model, X_test, y_test, eps)
adv_acc, adv_bypassed = get_metrics(model, X_adv, y_test)

# Dashboard Metrics
m1, m2, m3 = st.columns(3)
m1.metric("System Integrity (Normal)", f"{clean_acc:.1f}%")
m2.metric("System Integrity (Under Attack)", f"{adv_acc:.1f}%", f"{adv_acc - clean_acc:.1f}%")
m3.metric("Critical Threats Bypassed", f"{adv_bypassed}", f"+{adv_bypassed - clean_bypassed} breaches", delta_color="inverse")

st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 32px 0;'>", unsafe_allow_html=True)

# Visual Evidence
c1, c2 = st.columns(2)

with c1:
    st.plotly_chart(
        render_plot(X_test, y_test, model, "SECURE ENVIRONMENT", "AI border correctly blocking red threats."), 
        use_container_width=True, config={'displayModeBar': False}
    )

with c2:
    if eps == 0.0:
        st.plotly_chart(
            render_plot(X_adv, y_test, model, "ACTIVE THREAT SCENARIO", "Awaiting threat escalation from command panel."), 
            use_container_width=True, config={'displayModeBar': False}
        )
    else:
        st.plotly_chart(
            render_plot(X_adv, y_test, model, "SYSTEM COMPROMISED", "White arrows trace threats crossing into the safe zone.", arrows_from=X_test), 
            use_container_width=True, config={'displayModeBar': False}
        )

# Conclusion
st.markdown("""
<div class="glass-box" style="margin-top: 24px;">
    <h4 style="color: #FFFFFF; margin-top: 0;">STRATEGIC IMPLICATION</h4>
    <p style="margin-bottom: 0;">
        Watch the white trajectory lines on the right panel as you increase the threat level. The adversary barely alters the original profile. A human auditor would notice nothing wrong. Yet this minor mathematical shift completely breaks the AI. Deploying these models without adversarial stress testing guarantees a systemic vulnerability.
    </p>
</div>
""", unsafe_allow_html=True)
