"""
Interactive walkthrough: adversarial brittleness in AI systems.
Step-by-step guided experience for non-technical decision-makers.
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Brittleness · Walkthrough", layout="wide")

# ── state ───────────────────────────────────────────────────────────────────

if "step" not in st.session_state:
    st.session_state.step = 0
if "model" not in st.session_state:
    st.session_state.model = None
if "data" not in st.session_state:
    st.session_state.data = None

# ── css ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #FBFBF9;
    --card: #FFFFFF;
    --border: #EBEBEB;
    --text: #1D1D1F;
    --muted: #86868B;
    --accent: #0071E3;
    --accent-light: #EBF5FF;
    --green: #248A3D;
    --red: #D70015;
    --green-bg: rgba(36,138,61,0.08);
    --red-bg: rgba(215,0,21,0.08);
}

.stApp { background: var(--bg); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

* { font-family: 'Inter', -apple-system, sans-serif !important; }
h1, h2, h3 { color: var(--text); font-weight: 600 !important; }
p, li, span { color: var(--text); }

/* sidebar hidden — we use inline controls */
section[data-testid="stSidebar"] { display: none; }

/* cards */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 32px;
    margin: 16px 0;
}

/* step indicator */
.steps {
    display: flex;
    gap: 8px;
    align-items: center;
    margin: 0 0 32px 0;
}
.dot {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78rem; font-weight: 600;
}
.dot-active { background: var(--accent); color: white; }
.dot-done { background: var(--accent-light); color: var(--accent); }
.dot-future { background: var(--border); color: var(--muted); }
.step-line { width: 32px; height: 1px; background: var(--border); }
.step-label {
    font-size: 0.72rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--muted); margin-left: 10px;
}

/* metrics */
div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
}
div[data-testid="stMetric"] label {
    font-size: 0.72rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.04em;
    color: var(--muted) !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 2rem; font-weight: 600; color: var(--text) !important;
}

/* buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    padding: 0.5rem 1.5rem;
    font-size: 0.9rem;
}
.stButton > button[kind="primary"] {
    background: var(--accent); border: none; color: white;
}
.stButton > button[kind="primary"]:hover {
    background: #0077ED;
}
.stButton > button[kind="secondary"] {
    background: transparent; border: 1px solid var(--border); color: var(--text);
}

/* plotly toolbar hide */
.modebar { display: none !important; }

/* legend */
.legend {
    display: flex; gap: 16px; align-items: center;
    font-size: 0.8rem; color: var(--muted);
    margin: 8px 0 0 0;
}
.ldot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 5px; }
</style>
""", unsafe_allow_html=True)

# ── ml helpers ──────────────────────────────────────────────────────────────

class Net(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,2))
    def forward(self, x): return self.net(x)

def make_dataset(name, n, nz):
    if name=="Moons": X,y=make_moons(n_samples=n,noise=nz,random_state=42)
    elif name=="Circles": X,y=make_circles(n_samples=n,noise=nz,factor=0.5,random_state=42)
    else: X,y=make_classification(n_samples=n,n_features=2,n_redundant=0,n_informative=2,n_clusters_per_class=1,flip_y=nz,random_state=42)
    sc=StandardScaler(); X=sc.fit_transform(X); return X,y

def train_model(m, X, y, ep):
    o=torch.optim.Adam(m.parameters(),lr=0.01); L=nn.CrossEntropyLoss()
    Xt,yt=torch.tensor(X,dtype=torch.float32),torch.tensor(y,dtype=torch.long)
    for _ in range(ep): o.zero_grad(); l=L(m(Xt),yt); l.backward(); o.step()

def do_fgsm(m,X,y,e):
    Xt=torch.tensor(X,dtype=torch.float32,requires_grad=True); yt=torch.tensor(y,dtype=torch.long)
    nn.CrossEntropyLoss()(m(Xt),yt).backward(); return(Xt+e*Xt.grad.sign()).detach().numpy()

def do_pgd(m,X,y,e,s=10):
    Xt=torch.tensor(X,dtype=torch.float32); yt=torch.tensor(y,dtype=torch.long); Xa=Xt.clone(); a=e/4
    for _ in range(s):
        Xa.requires_grad_(True); nn.CrossEntropyLoss()(m(Xa),yt).backward()
        with torch.no_grad(): Xa=Xt+torch.clamp(Xa+a*Xa.grad.sign()-Xt,-e,e)
    return Xa.detach().numpy()

def do_noise(X,e): return X+np.random.uniform(-e,e,size=X.shape)

def make_grid(m,X,r=160):
    xs=np.linspace(X[:,0].min()-1,X[:,0].max()+1,r); ys=np.linspace(X[:,1].min()-1,X[:,1].max()+1,r)
    xx,yy=np.meshgrid(xs,ys)
    with torch.no_grad(): Z=m(torch.tensor(np.c_[xx.ravel(),yy.ravel()],dtype=torch.float32)).argmax(1).numpy().reshape(xx.shape)
    return xx,yy,Z

def accuracy(m,X,y):
    with torch.no_grad(): return(m(torch.tensor(X,dtype=torch.float32)).argmax(1).numpy()==y).mean()*100

GRN="#248A3D"; RED="#D70015"
GRN_BG="rgba(36,138,61,0.07)"; RED_BG="rgba(215,0,21,0.07)"

# ── plot helper ─────────────────────────────────────────────────────────────

def scatter_plot(X, y, model=None, title="", height=420, show_boundary=True):
    fig = go.Figure()
    if show_boundary and model is not None:
        xx,yy,Z = make_grid(model, X)
        fig.add_trace(go.Contour(
            x=np.linspace(xx.min(),xx.max(),Z.shape[1]),
            y=np.linspace(yy.min(),yy.max(),Z.shape[0]),
            z=Z, showscale=False,
            colorscale=[[0,GRN_BG],[1,RED_BG]],
            contours=dict(showlines=True,coloring="fill"),
            line=dict(width=1.2,color="rgba(0,0,0,0.10)"),
            hoverinfo="skip"))
    colors = [GRN if yi==0 else RED for yi in y]
    fig.add_trace(go.Scatter(
        x=X[:,0], y=X[:,1], mode="markers",
        marker=dict(color=colors, size=6, line=dict(width=0.6, color="white"), opacity=0.85),
        hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>"))
    fig.update_layout(
        height=height, showlegend=False, title=None,
        margin=dict(t=8,b=8,l=8,r=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=12))
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    return fig

# ── step indicator ──────────────────────────────────────────────────────────

def render_steps(current):
    labels = ["Data", "Train", "Attack", "Result"]
    dots = ""
    for i, label in enumerate(labels):
        if i < current: cls = "dot-done"
        elif i == current: cls = "dot-active"
        else: cls = "dot-future"
        dots += f'<div class="dot {cls}">{i+1}</div>'
        if i < len(labels) - 1:
            dots += '<div class="step-line"></div>'
    dots += f'<span class="step-label">{labels[current]}</span>'
    st.markdown(f'<div class="steps">{dots}</div>', unsafe_allow_html=True)

# ── step 0: intro ──────────────────────────────────────────────────────────

if st.session_state.step == 0:
    render_steps(0)

    st.markdown("## Every AI system has a breaking point.")
    st.markdown(
        "This walkthrough lets you find it. You'll build a small AI, "
        "then attack it — and see exactly how it fails."
    )

    st.markdown("")

    st.markdown("First, pick a dataset. These are simple 2D patterns the AI will learn to classify.")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        ds_choice = "Moons"
        fig_m = scatter_plot(*make_dataset("Moons", 300, 0.2), height=220, show_boundary=False)
        st.plotly_chart(fig_m, use_container_width=True, key="preview_moons")
        st.caption("**Moons** — two interlocking crescents")

    with c2:
        fig_c = scatter_plot(*make_dataset("Circles", 300, 0.2), height=220, show_boundary=False)
        st.plotly_chart(fig_c, use_container_width=True, key="preview_circles")
        st.caption("**Circles** — one group inside another")

    with c3:
        fig_b = scatter_plot(*make_dataset("Blobs", 300, 0.2), height=220, show_boundary=False)
        st.plotly_chart(fig_b, use_container_width=True, key="preview_blobs")
        st.caption("**Blobs** — two separated clusters")

    st.markdown('</div>', unsafe_allow_html=True)

    ds = st.selectbox("Choose one", ["Moons", "Circles", "Blobs"], label_visibility="collapsed")
    nz = st.slider("How noisy should the data be?", 0.05, 0.40, 0.18, 0.05,
                    help="More noise = messier separation between groups. Real-world data is always noisy.")

    st.markdown("")
    if st.button("Next →", type="primary"):
        X, y = make_dataset(ds, 400, nz)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
        st.session_state.data = {"X":X, "y":y, "Xtr":Xtr, "Xte":Xte, "ytr":ytr, "yte":yte, "ds":ds, "nz":nz}
        st.session_state.step = 1
        st.rerun()

# ── step 1: train ───────────────────────────────────────────────────────────

elif st.session_state.step == 1:
    render_steps(1)
    d = st.session_state.data

    st.markdown("## Now the AI learns to separate the two groups.")
    st.markdown(
        "It will draw a **decision boundary** — a line dividing "
        "the plane into two zones. Everything on one side gets "
        "classified as green, the other side as red."
    )

    hid = st.slider("How complex should the AI be?", 4, 64, 16, 4,
                     help="More neurons = more complex boundary. Too many can overfit. Too few can underfit.")
    ep = st.slider("How many training rounds?", 50, 500, 200, 50,
                    help="More rounds = the AI has more time to learn the pattern.")

    if st.button("Train the AI →", type="primary"):
        model = Net(hid)
        with st.spinner("Learning…"):
            train_model(model, d["Xtr"], d["ytr"], ep)
        model.eval()
        cs = accuracy(model, d["Xte"], d["yte"])
        st.session_state.model = model
        st.session_state.data["clean_acc"] = cs
        st.session_state.data["hid"] = hid
        st.session_state.step = 2
        st.rerun()

    # show data while they configure
    st.markdown("")
    st.markdown(f'<div class="legend"><span><span class="ldot" style="background:{GRN};"></span>Group A</span><span><span class="ldot" style="background:{RED};"></span>Group B</span></div>', unsafe_allow_html=True)
    fig = scatter_plot(d["X"], d["y"], height=400, show_boundary=False)
    st.plotly_chart(fig, use_container_width=True, key="step1_data")

# ── step 2: attack ──────────────────────────────────────────────────────────

elif st.session_state.step == 2:
    render_steps(2)
    d = st.session_state.data
    model = st.session_state.model

    st.markdown("## The AI is trained. Now try to break it.")
    st.markdown(
        f"It's scoring **{d['clean_acc']:.0f}%** on data it hasn't seen before. "
        f"That's the baseline. Now pick an attack method and see what happens."
    )

    c1, c2 = st.columns(2)
    with c1:
        atk = st.selectbox("Attack method", ["FGSM", "PGD", "Random noise"],
                           help="FGSM: fast single-step attack. PGD: stronger multi-step. Random noise: not targeted — a control.")
    with c2:
        eps = st.slider("Attack strength (ε)", 0.0, 2.0, 0.30, 0.05,
                        help="How far each point can be moved. Even small values can break the AI.")

    if st.button("Launch attack →", type="primary"):
        with st.spinner("Attacking…"):
            if atk == "FGSM": Xa = do_fgsm(model, d["Xte"], d["yte"], eps)
            elif atk == "PGD": Xa = do_pgd(model, d["Xte"], d["yte"], eps)
            else: Xa = do_noise(d["Xte"], eps)
        av = accuracy(model, Xa, d["yte"])
        st.session_state.data["Xa"] = Xa
        st.session_state.data["adv_acc"] = av
        st.session_state.data["atk"] = atk
        st.session_state.data["eps"] = eps
        st.session_state.step = 3
        st.rerun()

    # show trained model
    st.markdown("")
    st.markdown(f'<div class="legend"><span><span class="ldot" style="background:{GRN};"></span>Group A</span><span><span class="ldot" style="background:{RED};"></span>Group B</span><span style="margin-left:4px;">Shaded = AI decision zones · Line = boundary</span></div>', unsafe_allow_html=True)
    fig = scatter_plot(d["Xte"], d["yte"], model=model, height=440)
    st.plotly_chart(fig, use_container_width=True, key="step2_model")

# ── step 3: result ──────────────────────────────────────────────────────────

elif st.session_state.step == 3:
    render_steps(3)
    d = st.session_state.data
    model = st.session_state.model
    dr = d["clean_acc"] - d["adv_acc"]

    st.markdown("## Here's what happened.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Before", f"{d['clean_acc']:.0f}%")
    m2.metric("After", f"{d['adv_acc']:.0f}%")
    m3.metric("Lost", f"{dr:.0f} pts", delta=f"−{dr:.0f}", delta_color="inverse")

    # side by side plots
    st.markdown("")
    p1, p2 = st.columns(2)

    with p1:
        st.markdown(f'<div style="font-size:0.78rem; font-weight:500; color:#86868B; text-transform:uppercase; letter-spacing:0.06em;">Before attack</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="legend"><span><span class="ldot" style="background:{GRN};"></span>Group A</span><span><span class="ldot" style="background:{RED};"></span>Group B</span></div>', unsafe_allow_html=True)
        fig1 = scatter_plot(d["Xte"], d["yte"], model=model, height=400)
        st.plotly_chart(fig1, use_container_width=True, key="result_clean")

    with p2:
        st.markdown(f'<div style="font-size:0.78rem; font-weight:500; color:#86868B; text-transform:uppercase; letter-spacing:0.06em;">After {d["atk"]} attack · ε = {d["eps"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="legend"><span><span class="ldot" style="background:{GRN};"></span>Group A</span><span><span class="ldot" style="background:{RED};"></span>Group B</span><span style="margin-left:4px;">Arrows = how points moved</span></div>', unsafe_allow_html=True)

        # plot with arrows
        fig2 = scatter_plot(d["Xa"], d["yte"], model=model, height=400)
        rng = np.random.RandomState(0)
        idx = rng.choice(len(d["Xte"]), size=min(30, len(d["Xte"])), replace=False)
        for i in idx:
            fig2.add_annotation(
                x=d["Xa"][i,0], y=d["Xa"][i,1],
                ax=d["Xte"][i,0], ay=d["Xte"][i,1],
                showarrow=True, arrowhead=3, arrowsize=0.7, arrowwidth=0.6,
                arrowcolor="rgba(0,0,0,0.15)")
        st.plotly_chart(fig2, use_container_width=True, key="result_adv")

    # insight
    if dr > 15:
        msg = (f"A **{dr:.0f}-point drop**. The input barely changed — a human couldn't tell the "
               f"difference. But the AI's predictions collapsed. In a medical or financial system, "
               f"this would be a safety-critical failure.")
    elif dr > 5:
        msg = (f"**{dr:.0f} points** of accuracy gone. The attack needed no insider access to the "
               f"AI — just the ability to slightly modify inputs before they reach the system.")
    elif dr > 0.5:
        msg = (f"The overall drop was small (**{dr:.1f} pts**), but look at the arrows — individual "
               f"points crossed the boundary. An attacker only needs to flip *their* input.")
    else:
        msg = "The model held at this attack strength. Try going back and increasing ε, or switching to PGD."

    st.markdown("")
    st.info(msg)

    st.markdown("")
    st.markdown("### What this means")
    st.markdown(
        "The AI's confidence on normal data masks a hidden fragility. "
        "The decision boundary — the rule it learned — runs dangerously "
        "close to the data. A calculated nudge is enough to cross it. "
        "Most deployed AI systems have **never been tested** this way. "
        "No major jurisdiction currently requires it."
    )

    # actions
    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Try different attack settings", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with c2:
        if st.button("← Start over with new data", use_container_width=True):
            st.session_state.step = 0
            st.session_state.model = None
            st.session_state.data = None
            st.rerun()

# ── footer ──────────────────────────────────────────────────────────────────

st.markdown("")
st.markdown("")
st.caption("Prototype · Not for redistribution")
