import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Configuration & Core Styling
st.set_page_config(page_title="AI Vulnerability Assessment", layout="wide")

st.markdown("""
<style>
    /* Obsidian Dark Theme - Intelligence Dashboard Aesthetic */
    .stApp, header[data-testid="stHeader"] { background-color: #050505; color: #E5E5E5; }
    
    /* Completely remove sidebar and toggle */
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
    
    /* Typography */
    * { font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; }
    h1 { color: #FFFFFF !important; font-weight: 700 !important; font-size: 2.2rem !important; margin-bottom: 0.2rem !important; }
    h3 { color: #A3A3A3 !important; font-weight: 400 !important; font-size: 1.1rem !important; margin-bottom: 2rem !important; }
    p { font-size: 1.05rem; line-height: 1.6; color: #D4D4D8; }
    
    /* BLUF (Bottom Line Up Front) Box */
    .bluf-box {
        background-color: #121212;
        border-left: 4px solid #DC2626;
        padding: 20px 24px;
        margin-bottom: 32px;
        border-top: 1px solid #262626;
        border-right: 1px solid #262626;
        border-bottom: 1px solid #262626;
    }
    .bluf-text { color: #F5F5F5; font-weight: 500; font-size: 1.1rem; }
    
    /* Metric styling */
    div[data-testid="stMetric"] { background-color: #0A0A0A; border: 1px solid #262626; padding: 16px; border-radius: 2px; }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700 !important; }
    
    /* Custom Slider */
    .stSlider > div > div > div { color: #FFFFFF !important; font-weight: 600; font-size: 1.1rem; }
    
    /* Hide Plotly Toolbar */
    .modebar { display: none !important; }
</style>
""", unsafe_allow_html=True)

# 2. AI & Math Engine
C_SAFE = "#0EA5E9"       # Cerulean (Cleared)
C_THREAT = "#E11D48"     # Crimson (Threat)
C_SAFE_BG = "rgba(14, 165, 233, 0.08)"
C_THREAT_BG = "rgba(225, 29, 72, 0.08)"

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
        line=dict(width=1, color="rgba(255,255,255,0.15)"), hoverinfo="skip"))
    
    colors = [C_SAFE if yi == 0 else C_THREAT for yi in y]
    fig.add_trace(go.Scatter(
        x=X[:,0], y=X[:,1], mode="markers", hoverinfo="skip",
        marker=dict(color=colors, size=8, line=dict(width=1, color="#050505"))))
    
    if arrows_from is not None:
        for i in range(len(X)):
            if np.linalg.norm(X[i] - arrows_from[i]) > 0.01:
                fig.add_annotation(
                    x=X[i,0], y=X[i,1], ax=arrows_from[i,0], ay=arrows_from[i,1],
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="rgba(255, 255, 255, 0.8)")

    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:12px;color:#A3A3A3'>{subtitle}</span>", x=0.05, y=0.95),
        height=480, showlegend=False, margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

# 3. Intelligence Briefing Layout

st.title("Threat Assessment: AI Decision Boundaries")
st.markdown("### Classified Briefing on Adversarial Machine Learning Capabilities")

st.markdown("""
<div class="bluf-box">
    <div class="bluf-text">BOTTOM LINE UP FRONT (BLUF)</div>
    <p style="margin-top: 8px; margin-bottom: 0;">
        Artificial Intelligence systems do not "understand" data. They draw rigid mathematical borders between safe and dangerous activity. 
        Adversaries exploit this blind spot. By applying microscopic, calculated alterations to malicious data, an adversary can push their profile across the AI's border. 
        <strong>The AI will then wave the threat through, registering it as completely safe with 100% confidence.</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Process Data Live
X_train, X_test, y_train, y_test = get_data()
model = train_model(X_train, y_train)

# Slider acts as the main interactive element
st.markdown("<h4 style='color: #FFFFFF; margin-bottom: 10px;'>ESCALATE THREAT LEVEL</h4>", unsafe_allow_html=True)
eps = st.slider("Intensity of Adversarial Evasion Tactics", 0.0, 1.0, 0.0, 0.05, label_visibility="collapsed")
st.markdown("<br>", unsafe_allow_html=True)

# Compute Results
clean_acc, clean_bypassed = get_metrics(model, X_test, y_test)
X_adv = generate_attack(model, X_test, y_test, eps)
adv_acc, adv_bypassed = get_metrics(model, X_adv, y_test)

# Telemetry Metrics
m1, m2, m3 = st.columns(3)
m1.metric("System Integrity (Normal)", f"{clean_acc:.1f}%")
m2.metric("System Integrity (Under Attack)", f"{adv_acc:.1f}%", f"{adv_acc - clean_acc:.1f}%")
m3.metric("Critical Threats Bypassed", f"{adv_bypassed}", f"+{adv_bypassed - clean_bypassed} breaches", delta_color="inverse")

st.markdown("<hr style='border-color: #262626; margin: 32px 0;'>", unsafe_allow_html=True)

# Visual Evidence Panels
c1, c2 = st.columns(2)

with c1:
    st.plotly_chart(
        render_plot(X_test, y_test, model, "SECURE ENVIRONMENT", "AI border correctly blocking crimson threats."), 
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

# Executive Conclusion
st.markdown("""
<div style="background-color: #121212; padding: 24px; border: 1px solid #262626; margin-top: 24px;">
    <h4 style="color: #FFFFFF; margin-top: 0;">STRATEGIC IMPLICATION</h4>
    <p style="margin-bottom: 0;">
        When the threat level escalates above zero, note the white trajectory lines on the right panel. The adversary has barely altered the profile of the threat. To a human auditor, these anomalies are invisible. However, the mathematical shift is enough to subvert the AI. Deploying machine learning in national security, financial markets, or infrastructure without adversarial stress-testing guarantees an exploitable vulnerability.
    </p>
</div>
""", unsafe_allow_html=True)
