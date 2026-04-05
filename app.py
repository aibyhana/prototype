"""
Interactive briefing: Can you break this AI?
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

# ── config ──────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Can you break this AI?", layout="wide")

# ── css ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif&family=DM+Sans:wght@400;500;600&display=swap');

.stApp { background: #F7F6F2; }
header[data-testid="stHeader"] { background: transparent; }

h1, h2, h3 { font-family: 'Instrument Serif', Georgia, serif !important; color: #1a1a1a; font-weight: 400 !important; }
p, li, label, .stMarkdown, .stText, span, div { font-family: 'DM Sans', sans-serif !important; }

section[data-testid="stSidebar"] { background: #1C1C1C; }
section[data-testid="stSidebar"] * { color: #D4D0C8 !important; }
section[data-testid="stSidebar"] .stSlider label {
    font-size: 0.8rem; text-transform: uppercase;
    letter-spacing: 0.06em; color: #908E88 !important;
}

div[data-testid="stMetric"] {
    background: white; border: 1px solid #E8E5DE;
    border-radius: 8px; padding: 20px 24px;
}
div[data-testid="stMetric"] label {
    font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.06em; color: #999 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'Instrument Serif', serif !important;
    font-size: 2.4rem; color: #1a1a1a !important;
}

.stButton > button[kind="primary"] {
    background: #B83A14; border: none; color: white;
    font-family: 'DM Sans'; font-weight: 600; font-size: 0.95rem;
    border-radius: 6px; padding: 0.6rem 1.5rem;
}
.stButton > button[kind="primary"]:hover { background: #D14A1E; }

#MainMenu, footer { visibility: hidden; }

.hero {
    padding: 48px 0 32px 0;
    max-width: 640px;
}
.hero h1 {
    font-size: 2.8rem !important;
    line-height: 1.15;
    margin-bottom: 16px;
}
.hero p {
    font-size: 1.05rem;
    color: #666;
    line-height: 1.6;
}

.step-row {
    display: flex; gap: 24px; margin: 32px 0;
}
.step-card {
    flex: 1;
    background: white;
    border: 1px solid #E8E5DE;
    border-radius: 10px;
    padding: 28px 24px;
}
.step-num {
    font-family: 'Instrument Serif', serif;
    font-size: 2rem;
    color: #B83A14;
    line-height: 1;
    margin-bottom: 8px;
}
.step-card h4 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem;
    font-weight: 600;
    margin: 0 0 6px 0;
    color: #1a1a1a;
}
.step-card p {
    font-size: 0.85rem;
    color: #777;
    margin: 0;
    line-height: 1.5;
}

.insight-bar {
    background: #1C1C1C;
    border-radius: 10px;
    padding: 24px 32px;
    margin: 24px 0;
    color: #E8E5DE;
    font-size: 0.92rem;
    line-height: 1.6;
}
.insight-bar strong { color: #F0A07C; }

.plot-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #999;
    margin-bottom: 4px;
}

.divider { border: none; border-top: 1px solid #E5E2DB; margin: 40px 0 32px 0; }

.tag {
    display: inline-block;
    background: #B83A14;
    color: white;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 3px;
}

.legend-row {
    display: flex; gap: 20px; align-items: center;
    font-size: 0.82rem; color: #888; margin: 8px 0 4px 0;
}
.legend-dot {
    display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── ml code ─────────────────────────────────────────────────────────────────

class Net(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2,h), nn.ReLU(), nn.Linear(h,h), nn.ReLU(), nn.Linear(h,2))
    def forward(self, x): return self.net(x)

def gen(name, n, nz):
    if name == "Moons": X, y = make_moons(n_samples=n, noise=nz, random_state=42)
    elif name == "Circles": X, y = make_circles(n_samples=n, noise=nz, factor=0.5, random_state=42)
    else: X, y = make_classification(n_samples=n, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, flip_y=nz, random_state=42)
    sc = StandardScaler(); X = sc.fit_transform(X); return X, y, sc

def trn(m, X, y, ep):
    o=torch.optim.Adam(m.parameters(),lr=0.01); L=nn.CrossEntropyLoss()
    Xt,yt=torch.tensor(X,dtype=torch.float32),torch.tensor(y,dtype=torch.long)
    for _ in range(ep): o.zero_grad(); l=L(m(Xt),yt); l.backward(); o.step()

def fgsm(m,X,y,e):
    Xt=torch.tensor(X,dtype=torch.float32,requires_grad=True); yt=torch.tensor(y,dtype=torch.long)
    nn.CrossEntropyLoss()(m(Xt),yt).backward(); return(Xt+e*Xt.grad.sign()).detach().numpy()

def pgd(m,X,y,e,s):
    Xt=torch.tensor(X,dtype=torch.float32); yt=torch.tensor(y,dtype=torch.long); Xa=Xt.clone(); a=e/4
    for _ in range(s):
        Xa.requires_grad_(True); nn.CrossEntropyLoss()(m(Xa),yt).backward()
        with torch.no_grad(): Xa=Xt+torch.clamp(Xa+a*Xa.grad.sign()-Xt,-e,e)
    return Xa.detach().numpy()

def noise(X,e): return X+np.random.uniform(-e,e,size=X.shape)

def grd(m,X,r=160):
    xs=np.linspace(X[:,0].min()-1,X[:,0].max()+1,r); ys=np.linspace(X[:,1].min()-1,X[:,1].max()+1,r)
    xx,yy=np.meshgrid(xs,ys)
    with torch.no_grad(): Z=m(torch.tensor(np.c_[xx.ravel(),yy.ravel()],dtype=torch.float32)).argmax(1).numpy().reshape(xx.shape)
    return xx,yy,Z

def sc(m,X,y):
    with torch.no_grad(): return(m(torch.tensor(X,dtype=torch.float32)).argmax(1).numpy()==y).mean()*100

G="#2D6A4F"; R="#9B2226"; GB="rgba(45,106,79,0.10)"; RB="rgba(155,34,38,0.10)"

# ── sidebar (minimal) ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Settings")
    ds = st.selectbox("Data shape", ["Moons", "Circles", "Blobs"])
    ns = st.slider("Samples", 200, 800, 400, 100)
    nz = st.slider("Noise", 0.05, 0.40, 0.20, 0.05)
    st.markdown("---")
    hid = st.slider("Model neurons", 4, 64, 16, 4)
    ep = st.slider("Training rounds", 50, 500, 200, 50)
    st.markdown("---")
    atk = st.selectbox("Attack", ["FGSM", "PGD", "Random noise"])
    eps = st.slider("Attack strength (ε)", 0.0, 2.0, 0.30, 0.05)
    st.markdown("---")
    run = st.button("Run", type="primary", use_container_width=True)

# ── landing ─────────────────────────────────────────────────────────────────

if not run:
    st.markdown("""
    <div class="hero">
        <span class="tag">Interactive Briefing</span>
        <h1>Can you break this AI?</h1>
        <p>
            You'll train a small AI system, then try to fool it with an
            invisible attack. It takes 30 seconds. No coding required.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="step-row">
        <div class="step-card">
            <div class="step-num">1</div>
            <h4>Data appears</h4>
            <p>Two groups of dots — like "approve" vs "reject" in a loan system.</p>
        </div>
        <div class="step-card">
            <div class="step-num">2</div>
            <h4>AI learns a rule</h4>
            <p>It draws a line separating the groups. You'll see how accurate it is.</p>
        </div>
        <div class="step-card">
            <div class="step-num">3</div>
            <h4>You attack it</h4>
            <p>A tiny, invisible nudge to the data. Watch the accuracy collapse.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-bar">
        This demonstrates <strong>adversarial brittleness</strong> — a vulnerability
        present in every major AI architecture deployed today. Most systems in use
        have never been tested for it.
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ── experiment ──────────────────────────────────────────────────────────────

X,y,_=gen(ds,ns,nz)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=42)

model=Net(hid)
with st.spinner("Training…"): trn(model,Xtr,ytr,ep)
model.eval()
cs=sc(model,Xte,yte)

with st.spinner("Attacking…"):
    if atk=="FGSM": Xa=fgsm(model,Xte,yte,eps)
    elif atk=="PGD": Xa=pgd(model,Xte,yte,eps,10)
    else: Xa=noise(Xte,eps)

av=sc(model,Xa,yte); dr=cs-av

# ── metrics ─────────────────────────────────────────────────────────────────

st.markdown("")
c1,c2,c3=st.columns(3)
c1.metric("Before attack", f"{cs:.0f}%")
c2.metric("After attack", f"{av:.0f}%")
c3.metric("Accuracy lost", f"{dr:.0f} pts", delta=f"−{dr:.0f}", delta_color="inverse")

# ── plots ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="legend-row">
    <span><span class="legend-dot" style="background:#2D6A4F;"></span>Group A</span>
    <span><span class="legend-dot" style="background:#9B2226;"></span>Group B</span>
    <span style="margin-left:8px;">Shaded region = AI's decision zone · Thin line = decision boundary · Arrows = how points moved</span>
</div>
""", unsafe_allow_html=True)

xx,yy,Z=grd(model,X)
cp=[G if yi==0 else R for yi in yte]

fig=make_subplots(rows=1,cols=2,
    subplot_titles=("Clean data",f"After {atk} attack · ε={eps}"),
    horizontal_spacing=0.06)

for c in [1,2]:
    fig.add_trace(go.Contour(
        x=np.linspace(xx.min(),xx.max(),Z.shape[1]),
        y=np.linspace(yy.min(),yy.max(),Z.shape[0]),
        z=Z,showscale=False,
        colorscale=[[0,GB],[1,RB]],
        contours=dict(showlines=True,coloring="fill"),
        line=dict(width=1.5,color="rgba(0,0,0,0.15)"),
        hoverinfo="skip"),row=1,col=c)

fig.add_trace(go.Scatter(x=Xte[:,0],y=Xte[:,1],mode="markers",
    marker=dict(color=cp,size=6,line=dict(width=0.5,color="white")),
    hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>"),row=1,col=1)

fig.add_trace(go.Scatter(x=Xa[:,0],y=Xa[:,1],mode="markers",
    marker=dict(color=cp,size=6,line=dict(width=0.5,color="white")),
    hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>"),row=1,col=2)

rng=np.random.RandomState(0)
idx=rng.choice(len(Xte),size=min(30,len(Xte)),replace=False)
for i in idx:
    fig.add_annotation(x=Xa[i,0],y=Xa[i,1],ax=Xte[i,0],ay=Xte[i,1],
        xref="x2",yref="y2",axref="x2",ayref="y2",
        showarrow=True,arrowhead=3,arrowsize=0.7,arrowwidth=0.6,
        arrowcolor="rgba(0,0,0,0.18)")

fig.update_layout(height=520,showlegend=False,
    margin=dict(t=32,b=12,l=12,r=12),
    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans",size=12))
fig.update_xaxes(showgrid=False,zeroline=False,showticklabels=False)
fig.update_yaxes(showgrid=False,zeroline=False,showticklabels=False)
fig.update_annotations(font=dict(size=13,family="DM Sans"))

st.plotly_chart(fig,use_container_width=True)

# ── insight ─────────────────────────────────────────────────────────────────

if dr > 15:
    msg = f"The AI went from <strong>{cs:.0f}%</strong> to <strong>{av:.0f}%</strong> accuracy. The input barely changed. The model didn't change. But the predictions collapsed. In a medical or financial system, this is a safety failure."
elif dr > 5:
    msg = f"A <strong>{dr:.0f}-point</strong> drop. The attack didn't need insider access — just the ability to slightly modify inputs. Any system making real decisions at this error rate is vulnerable."
elif dr > 0.5:
    msg = f"Only <strong>{dr:.1f} points</strong> lost overall — but individual predictions flipped. An attacker only needs to fool the system on <em>their</em> input, not all of them."
else:
    msg = "The model held at this strength. Try increasing ε or switching from random noise to FGSM/PGD."

st.markdown(f'<div class="insight-bar">{msg}</div>', unsafe_allow_html=True)

# ── what to try ─────────────────────────────────────────────────────────────

st.markdown('<hr class="divider">', unsafe_allow_html=True)

t1,t2,t3=st.columns(3)
with t1:
    st.markdown("**↑ Increase ε**")
    st.caption("Watch accuracy collapse. At what point is the failure unacceptable?")
with t2:
    st.markdown("**Switch to PGD**")
    st.caption("Compare a calculated attack vs random noise. The difference is striking.")
with t3:
    st.markdown("**Fewer neurons**")
    st.caption("Simpler models resist some attacks better — but also make more mistakes.")

# ── footer ──────────────────────────────────────────────────────────────────

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.caption("Prototype · Not for redistribution")
