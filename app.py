import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Premium Dribbble-Style SaaS Configuration
st.set_page_config(page_title="AI Robustness Sandbox", layout="wide", initial_sidebar_state="collapsed")

# Inject Custom CSS, Google Fonts, and modern UI styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    /* Base App Styling */
    .stApp { 
        background-color: #F8FAFC; 
        color: #0F172A;
    }
    
    * { font-family: 'Plus Jakarta Sans', sans-serif !important; }
    
    /* Hide Streamlit Clutter */
    header[data-testid="stHeader"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }[data-testid="stSidebar"] { display: none !important; }
    .block-container { padding-top: 4rem !important; max-width: 1280px !important; }
    
    /* Hero Section */
    .hero-title {
        font-weight: 800;
        font-size: 3.2rem;
        line-height: 1.1;
        letter-spacing: -0.03em;
        color: #0F172A;
        margin-bottom: 1.5rem;
    }
    .hero-title span {
        background: linear-gradient(135deg, #4F46E5 0%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-text {
        font-size: 1.15rem;
        line-height: 1.6;
        color: #475569;
        margin-bottom: 2rem;
        max-width: 90%;
    }
    .hero-image {
        width: 100%;
        height: 380px;
        object-fit: cover;
        border-radius: 24px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
    }
    
    /* Control Panel Card */
    .control-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 32px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    
    /* Metrics Override */
    div[data-testid="stMetric"] { 
        background-color: #FFFFFF; 
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    }
    div[data-testid="stMetricLabel"] { color: #64748B !important; font-weight: 600 !important; font-size: 0.85rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
    div[data-testid="stMetricValue"] { color: #0F172A !important; font-weight: 800 !important; font-size: 2.5rem !important; letter-spacing: -0.02em; }
    
    /* Badge */
    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-bottom: 1rem;
        background-color: #EEF2FF;
        color: #4F46E5;
        border: 1px solid #C7D2FE;
    }
</style>
""", unsafe_allow_html=True)

# 2. ML Engine (Hidden away from the UI logic)
C_SAFE = "#4F46E5"       # Indigo
C_THREAT = "#EC4899"     # Rose
C_SAFE_BG = "#EEF2FF"
C_THREAT_BG = "#FDF2F8"

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
    
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=Z, showscale=False, colorscale=[[0, C_SAFE_BG],[1, C_THREAT_BG]],
        contours=dict(showlines=False, coloring="fill"), hoverinfo="skip"))
    
    colors =[C_SAFE if yi == 0 else C_THREAT for yi in y]
    fig.add_trace(go.Scatter(
        x=X[:,0], y=X[:,1], mode="markers", hoverinfo="skip",
        marker=dict(color=colors, size=9, line=dict(width=2, color="#FFFFFF"), opacity=0.9)))
    
    if arrows_from is not None:
        for i in range(len(X)):
            if np.linalg.norm(X[i] - arrows_from[i]) > 0.01:
                fig.add_annotation(
                    x=X[i,0], y=X[i,1], ax=arrows_from[i,0], ay=arrows_from[i,1],
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#94A3B8", opacity=0.8)

    fig.update_layout(
        title=dict(text=f"<span style='font-size:20px; font-weight:700; color:#0F172A; font-family:\"Plus Jakarta Sans\"'>{title}</span><br><span style='font-size:14px;color:#64748B; font-weight:500; font-family:\"Plus Jakarta Sans\"'>{subtitle}</span>", x=0.02, y=0.95),
        height=450, showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

# 3. UI Layout

# Hero Section
col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.markdown("<div class='status-badge'>Prototype Demo</div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='hero-title'>Why regulators should care about <span>robustness.</span></div>
        <div class='hero-text'>
            There is a dangerous gap between what AI systems actually do and what non-technical regulators think they do. 
            Models don't understand context—they draw invisible mathematical lines. By making microscopic numerical tweaks to a data point, adversaries can seamlessly bypass safety guardrails.
            <br><br>
            <strong>Experience it yourself below.</strong> Watch how a perfectly accurate classifier collapses when subjected to minor adversarial pressure.
        </div>
    """, unsafe_allow_html=True)

with col2:
    # High-quality abstract tech image from Unsplash (representing boundaries/data)
    st.markdown("""
        <img src="https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=1200&q=80" class="hero-image" alt="Abstract AI Data">
    """, unsafe_allow_html=True)


# Process Models
X_train, X_test, y_train, y_test = get_data()
model = train_model(X_train, y_train)

# Interactive Control Panel Card
st.markdown("<div class='control-card'>", unsafe_allow_html=True)
st.markdown("<h4 style='margin-top:0; color:#0F172A;'>Interactive Threat Simulator</h4>", unsafe_allow_html=True)

col_slider, col_desc = st.columns([1, 1.5])
with col_slider:
    eps = st.slider("Adversarial Perturbation (Epsilon)", 0.0, 1.0, 0.0, 0.05)
with col_desc:
    if eps == 0.0:
        msg = "<b>Clean Data:</b> The model operates exactly as intended, evaluating standard, unaltered data."
    elif eps <= 0.35:
        msg = f"<b>Mild Attack (ε={eps:.2f}):</b> Invisible adjustments. Real-world equivalent: Changing a few invisible pixels in an image to bypass a safety filter."
    elif eps <= 0.7:
        msg = f"<b>Moderate Attack (ε={eps:.2f}):</b> Real-world equivalent: Placing a small sticker on a stop sign to confuse an autonomous vehicle's vision system."
    else:
        msg = f"<b>Severe Attack (ε={eps:.2f}):</b> Real-world equivalent: Heavy digital noise applied to a financial transaction to force automated approval."
    st.info(msg, icon="💡")

st.markdown("</div>", unsafe_allow_html=True)

# Metrics & Computations
clean_acc, clean_bypassed = get_metrics(model, X_test, y_test)
X_adv = generate_attack(model, X_test, y_test, eps)
adv_acc, adv_bypassed = get_metrics(model, X_adv, y_test)

m1, m2, m3 = st.columns(3)
m1.metric("Baseline Accuracy", f"{clean_acc:.1f}%")
m2.metric("Accuracy Under Attack", f"{adv_acc:.1f}%", f"{adv_acc - clean_acc:.1f}%")
m3.metric("Safety Guardrails Bypassed", f"{adv_bypassed}", f"+{adv_bypassed - clean_bypassed} failures", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# Charts Section
c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown("<div style='border: 1px solid #E2E8F0; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);'>", unsafe_allow_html=True)
    st.plotly_chart(render_chart(X_test, y_test, model, "Expected Behavior", "Baseline classification on clean data."), use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div style='border: 1px solid #E2E8F0; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);'>", unsafe_allow_html=True)
    if eps == 0.0:
        st.plotly_chart(render_chart(X_adv, y_test, model, "Adversarial Behavior", "Increase perturbation to view system collapse."), use_container_width=True, config={'displayModeBar': False})
    else:
        st.plotly_chart(render_chart(X_adv, y_test, model, "System Collapse", "Gray vectors trace data forced across the boundary."), use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)
