import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Dribbble-Inspired SaaS Configuration & CSS
st.set_page_config(page_title="Model Robustness Explorer", layout="wide")

st.markdown("""
<style>
    /* Clean Light SaaS Background */
    .stApp { 
        background-color: #F8FAFC; 
        color: #0F172A;
    }
    
    /* Hide default Streamlit clutter */
    header[data-testid="stHeader"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
    .block-container { padding-top: 3rem !important; max-width: 1200px !important; }
    
    /* Modern Typography */
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
    h1 { color: #0F172A !important; font-weight: 800 !important; font-size: 2.5rem !important; letter-spacing: -0.02em !important; margin-bottom: 0.2rem !important; }
    h3 { color: #64748B !important; font-weight: 400 !important; font-size: 1.2rem !important; margin-bottom: 2rem !important; }
    p { font-size: 1.05rem; line-height: 1.6; color: #475569; }
    
    /* Clean Dribbble-style Cards */
    .info-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 28px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        margin-bottom: 32px;
    }
    
    /* Context Banner */
    .context-banner {
        background: linear-gradient(to right, #EEF2FF, #F8FAFC);
        border-left: 4px solid #4F46E5;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin-bottom: 24px;
        color: #3730A3;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    /* Metric Cards Styling */
    div[data-testid="stMetric"] { 
        background-color: #FFFFFF; 
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 20px 24px; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    }
    div[data-testid="stMetricLabel"] { color: #64748B !important; font-weight: 600 !important; font-size: 0.85rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
    div[data-testid="stMetricValue"] { color: #0F172A !important; font-weight: 800 !important; font-size: 2.2rem !important; }
    div[data-testid="stMetricDelta"] svg { stroke-width: 3px !important; }
    
    /* Hide Plotly Toolbar */
    .modebar { display: none !important; }
</style>
""", unsafe_allow_html=True)

# 2. Mathematical Engine & AI Setup
# Modern UI Colors
C_SAFE = "#4F46E5"       # Indigo
C_THREAT = "#EC4899"     # Rose
C_SAFE_BG = "#EEF2FF"    # Very Light Indigo
C_THREAT_BG = "#FDF2F8"  # Very Light Rose

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
    
    # Soft background contours
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=Z, showscale=False, colorscale=[[0, C_SAFE_BG], [1, C_THREAT_BG]],
        contours=dict(showlines=False, coloring="fill"), hoverinfo="skip"))
    
    # Clean data points with white borders
    colors =[C_SAFE if yi == 0 else C_THREAT for yi in y]
    fig.add_trace(go.Scatter(
        x=X[:,0], y=X[:,1], mode="markers", hoverinfo="skip",
        marker=dict(color=colors, size=9, line=dict(width=1.5, color="#FFFFFF"), opacity=0.9)))
    
    # Subtle arrows for adversarial shifts
    if arrows_from is not None:
        for i in range(len(X)):
            if np.linalg.norm(X[i] - arrows_from[i]) > 0.01:
                fig.add_annotation(
                    x=X[i,0], y=X[i,1], ax=arrows_from[i,0], ay=arrows_from[i,1],
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#64748B", opacity=0.7)

    fig.update_layout(
        title=dict(text=f"<span style='font-size:18px; font-weight:700; color:#0F172A; font-family:Inter'>{title}</span><br><span style='font-size:13px;color:#64748B; font-family:Inter'>{subtitle}</span>", x=0.02, y=0.95),
        height=450, showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

# 3. UI Layout and Content

st.title("Adversarial AI Explorer")
st.markdown("### Visualizing how subtle data manipulation breaks ML decision boundaries.")

st.markdown("""
<div class="info-card">
    <h4 style="color: #0F172A; margin-top: 0; margin-bottom: 8px;">Why Regulators Care About Robustness</h4>
    <p style="margin: 0;">
        Machine learning models don't "see" data the way we do; they simply draw mathematical boundaries between categories. 
        Adversarial attacks exploit this. By making mathematically calculated, microscopic tweaks to a data point, 
        an attacker can easily push a malicious profile across the boundary into the safe zone. The AI will then approve it with absolute confidence.
    </p>
</div>
""", unsafe_allow_html=True)

# Process Data Live
X_train, X_test, y_train, y_test = get_data()
model = train_model(X_train, y_train)

# Interactive Component
st.markdown("<p style='font-weight: 600; color: #0F172A; margin-bottom: -10px;'>Adjust Perturbation Strength (Epsilon)</p>", unsafe_allow_html=True)
eps = st.slider("Intensity of Adversarial Evasion Tactics", 0.0, 1.0, 0.0, 0.05, label_visibility="collapsed")

# Dynamic Context Updates
if eps == 0.0:
    example_text = "Clean Data (Epsilon 0.0). The model is evaluating standard, unaltered data."
elif eps <= 0.35:
    example_text = f"Mild Perturbation (Epsilon {eps:.2f}). Real-world equivalent: Changing a few invisible pixels in an image to bypass a content filter."
elif eps <= 0.7:
    example_text = f"Moderate Perturbation (Epsilon {eps:.2f}). Real-world equivalent: Placing a small, specific sticker on a stop sign to confuse a self-driving car."
else:
    example_text = f"Severe Perturbation (Epsilon {eps:.2f}). Real-world equivalent: Heavy digital noise applied to an X-ray to force an automated misdiagnosis."

st.markdown(f"""
<div class="context-banner">
    <strong>Current Simulation:</strong> {example_text}
</div>
""", unsafe_allow_html=True)

# Computations
clean_acc, clean_bypassed = get_metrics(model, X_test, y_test)
X_adv = generate_attack(model, X_test, y_test, eps)
adv_acc, adv_bypassed = get_metrics(model, X_adv, y_test)

# Dashboard Metrics
m1, m2, m3 = st.columns(3)
m1.metric("Baseline Accuracy", f"{clean_acc:.1f}%")
m2.metric("Accuracy Under Attack", f"{adv_acc:.1f}%", f"{adv_acc - clean_acc:.1f}%")
m3.metric("False Negatives (Bypassed)", f"{adv_bypassed}", f"+{adv_bypassed - clean_bypassed} errors", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)

# Visualizations
c1, c2 = st.columns(2)

with c1:
    st.plotly_chart(
        render_plot(X_test, y_test, model, "Baseline Model (Clean Data)", "The AI correctly separates blue (safe) and pink (unsafe) data."), 
        use_container_width=True, config={'displayModeBar': False}
    )

with c2:
    if eps == 0.0:
        st.plotly_chart(
            render_plot(X_adv, y_test, model, "Adversarial Evaluation", "Increase the slider above to apply perturbation."), 
            use_container_width=True, config={'displayModeBar': False}
        )
    else:
        st.plotly_chart(
            render_plot(X_adv, y_test, model, "Decision Boundary Shift", "Gray arrows show how pink data points are forced into the blue zone."), 
            use_container_width=True, config={'displayModeBar': False}
        )
