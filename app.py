import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. PAGE CONFIG & HIGH-END CSS
# ==========================================
st.set_page_config(page_title="AI Robustness", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Import modern system fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Clean, ultra-minimalist background */
    .stApp { background-color: #FAFAFA; color: #18181B; font-family: 'Inter', sans-serif; }
    
    /* Remove all Streamlit clutter */
    header { visibility: hidden !important; }
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    .block-container { padding-top: 3rem !important; padding-bottom: 2rem !important; max-width: 1100px !important; }
    
    /* Typography Polish */
    h1, h2, h3 { font-family: 'Inter', sans-serif !important; letter-spacing: -0.02em !important; color: #09090B !important; }
    h1 { font-size: 2.5rem !important; font-weight: 700 !important; margin-bottom: 0.5rem !important; }
    p { font-size: 1.05rem; color: #52525B; line-height: 1.6; }
    
    /* Policy Briefing Blockquote */
    .policy-quote {
        background-color: #FFFFFF;
        border-left: 4px solid #000000;
        padding: 24px 32px;
        margin: 24px 0 40px 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        font-size: 1.1rem;
        color: #3F3F46;
        line-height: 1.7;
    }
    
    /* Beautiful Metric Cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E4E4E7;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    [data-testid="stMetricLabel"] { font-size: 0.85rem !important; font-weight: 600 !important; color: #71717A !important; text-transform: uppercase; letter-spacing: 0.05em; }
    [data-testid="stMetricValue"] { font-size: 2.2rem !important; font-weight: 700 !important; color: #09090B !important; }
    
    /* Slider Customization */
    .stSlider label { font-weight: 600 !important; color: #18181B !important; font-size: 1rem !important; }
    
    /* Context Box */
    .context-box {
        background-color: #F4F4F5;
        border-radius: 8px;
        padding: 16px 24px;
        color: #27272A;
        font-size: 0.95rem;
        height: 100%;
        display: flex;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. AI MATH ENGINE & COLORS
# ==========================================
C_SAFE = "#2563EB"       # Deep Blue
C_THREAT = "#DC2626"     # Crimson Red
C_SAFE_BG = "#EFF6FF"    # Very Light Blue
C_THREAT_BG = "#FEF2F2"  # Very Light Red

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

def render_chart(X, y, model, title, arrows_from=None):
    fig = go.Figure()
    xs = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 120)
    ys = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 120)
    xx, yy = np.meshgrid(xs, ys)
    with torch.no_grad():
        Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)).argmax(1).numpy().reshape(xx.shape)
    
    # Background contour (the AI's mental map)
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=Z, showscale=False, colorscale=[[0, C_SAFE_BG],[1, C_THREAT_BG]],
        contours=dict(showlines=False, coloring="fill"), hoverinfo="skip"))
    
    # Data points with elegant white borders
    colors =[C_SAFE if yi == 0 else C_THREAT for yi in y]
    fig.add_trace(go.Scatter(
        x=X[:,0], y=X[:,1], mode="markers", hoverinfo="skip",
        marker=dict(color=colors, size=9, line=dict(width=1.5, color="#FFFFFF"), opacity=0.9)))
    
    # Attack vectors (arrows)
    if arrows_from is not None:
        for i in range(len(X)):
            if np.linalg.norm(X[i] - arrows_from[i]) > 0.01:
                fig.add_annotation(
                    x=X[i,0], y=X[i,1], ax=arrows_from[i,0], ay=arrows_from[i,1],
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#71717A", opacity=0.8)

    # Clean styling
    fig.update_layout(
        title=dict(text=f"<b style='color:#09090B; font-size:18px;'>{title}</b>", x=0.03, y=0.95),
        height=400, showlegend=False, margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    # Add a clean border around the entire plot area
    fig.update_xaxes(showline=True, linewidth=1, linecolor='#E4E4E7', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='#E4E4E7', mirror=True)
    return fig

# ==========================================
# 3. UI LAYOUT & CONTENT
# ==========================================

st.title("Why Regulators Should Care About AI Robustness")

st.markdown("""
<div class="policy-quote">
    <strong>The Regulatory Gap:</strong> AI models do not possess human context—they simply draw mathematical boundaries based on data. 
    Because of this, bad actors can make microscopic, invisible adjustments to an input that seamlessly bypass safety guardrails. 
    To regulate AI effectively, we must first understand how easily these systems break.
</div>
""", unsafe_allow_html=True)

# Model Pipeline
X_train, X_test, y_train, y_test = get_data()
model = train_model(X_train, y_train)

# Interactive Section (Native Streamlit columns for perfect alignment)
col_control, col_text = st.columns([1, 1.2], gap="large")

with col_control:
    st.markdown("<p style='font-weight:600; margin-bottom:-10px;'>Adversarial Attack Strength</p>", unsafe_allow_html=True)
    eps = st.slider("Adversarial Attack Strength", 0.0, 1.0, 0.0, 0.05, label_visibility="collapsed")

with col_text:
    if eps == 0.0:
        desc = "<b>Clean Data:</b> The system is operating normally. Guardrails are intact."
    elif eps <= 0.35:
        desc = f"<b>Mild Perturbation (ε={eps:.2f}):</b> Invisible adjustments. Real-world equivalent: Changing a few pixels in an image to bypass an automated content filter."
    elif eps <= 0.7:
        desc = f"<b>Moderate Attack (ε={eps:.2f}):</b> Physical interference. Real-world equivalent: Placing a small piece of tape on a stop sign to confuse a self-driving car."
    else:
        desc = f"<b>Severe Attack (ε={eps:.2f}):</b> Complete distortion. Real-world equivalent: Applying digital noise to a financial transaction to force automated approval."
    
    st.markdown(f"<div class='context-box'>💡 &nbsp; {desc}</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Metrics Pipeline
clean_acc, clean_bypassed = get_metrics(model, X_test, y_test)
X_adv = generate_attack(model, X_test, y_test, eps)
adv_acc, adv_bypassed = get_metrics(model, X_adv, y_test)

m1, m2, m3 = st.columns(3)
m1.metric("Baseline Accuracy", f"{clean_acc:.1f}%")
m2.metric("Accuracy Under Attack", f"{adv_acc:.1f}%", f"{adv_acc - clean_acc:.1f}%")
m3.metric("Safety Guardrails Bypassed", f"{adv_bypassed}", f"+{adv_bypassed - clean_bypassed} critical failures", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# Visualizations
c1, c2 = st.columns(2, gap="large")

with c1:
    st.plotly_chart(
        render_chart(X_test, y_test, model, "Expected Behavior (Baseline)"), 
        use_container_width=True, 
        config={'displayModeBar': False}
    )

with c2:
    if eps == 0.0:
        st.plotly_chart(
            render_chart(X_adv, y_test, model, "Adversarial Behavior (Awaiting Input)"), 
            use_container_width=True, 
            config={'displayModeBar': False}
        )
    else:
        st.plotly_chart(
            render_chart(X_adv, y_test, model, "System Collapse (Guardrails Bypassed)", arrows_from=X_test), 
            use_container_width=True, 
            config={'displayModeBar': False}
        )

st.markdown("""
<p style="text-align:center; font-size:0.9rem; color:#A1A1AA; margin-top: 40px;">
    <i>Interactive Prototype — Designed for Policy & Compliance Briefings</i>
</p>
""", unsafe_allow_html=True)
