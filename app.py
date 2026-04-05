"""
Interactive briefing: Can AI systems be tricked?
Guided walkthrough for senior decision-makers. No technical background needed.
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Can AI Systems Be Tricked?", layout="wide")

if "step" not in st.session_state: st.session_state.step = 0
if "model" not in st.session_state: st.session_state.model = None
if "data" not in st.session_state: st.session_state.data = None

# ── css ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg: #0F1117;
    --surface: #181B22;
    --surface2: #1E222B;
    --border: #2A2E38;
    --text: #E4E4E7;
    --muted: #71717A;
    --accent: #6C9FFF;
    --accent2: #4F7ADB;
    --green: #4ADE80;
    --red: #F87171;
    --green-dim: rgba(74,222,128,0.10);
    --red-dim: rgba(248,113,113,0.10);
}

.stApp { background: var(--bg); color: var(--text); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }

* { font-family: 'Inter', -apple-system, sans-serif !important; }
h1, h2, h3 { color: var(--text) !important; font-weight: 600 !important; }
p, li, span, label { color: var(--text) !important; }

/* metric cards */
div[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px 24px;
}
div[data-testid="stMetric"] label {
    font-size: 0.7rem; font-weight: 500; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--muted) !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 2rem; font-weight: 600; color: var(--text) !important;
}

/* buttons */
.stButton > button[kind="primary"] {
    background: var(--accent); border: none; color: #0F1117;
    font-weight: 600; border-radius: 8px; padding: 0.55rem 1.6rem;
}
.stButton > button[kind="primary"]:hover { background: var(--accent2); color: white; }
.stButton > button[kind="secondary"],
.stButton > button:not([kind="primary"]) {
    background: var(--surface); border: 1px solid var(--border);
    color: var(--text); border-radius: 8px;
}

/* slider / select overrides for dark */
.stSlider > div > div > div { color: var(--text) !important; }
.stSelectbox > div > div { background: var(--surface); border-color: var(--border); color: var(--text); }

/* step dots */
.steps { display:flex; gap:8px; align-items:center; margin:0 0 36px 0; }
.dot {
    width:30px; height:30px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:0.75rem; font-weight:600;
}
.dot-on { background:var(--accent); color:#0F1117; }
.dot-done { background:var(--surface2); color:var(--accent); border:1px solid var(--accent); }
.dot-off { background:var(--surface2); color:var(--muted); border:1px solid var(--border); }
.sline { width:28px; height:1px; background:var(--border); }
.slabel { font-size:0.7rem; font-weight:500; text-transform:uppercase; letter-spacing:0.06em; color:var(--muted); margin-left:10px; }

/* context box — the plain english explanation */
.context {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 24px 28px;
    margin: 0 0 28px 0;
    line-height: 1.7;
    font-size: 0.92rem;
    color: #B4B4BB;
}
.context strong { color: var(--text); }
.context em { color: var(--accent); font-style: normal; }

/* legend */
.legend { display:flex; gap:16px; align-items:center; font-size:0.78rem; color:var(--muted); margin:8px 0 0 0; }
.ldot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:5px; }

/* plotly toolbar */
.modebar { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── ml ──────────────────────────────────────────────────────────────────────

class Net(nn.Module):
    def __init__(s,h):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(2,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,2))
    def forward(s,x): return s.net(x)

def mkdata(name,n,nz):
    if name=="Moons": X,y=make_moons(n_samples=n,noise=nz,random_state=42)
    elif name=="Circles": X,y=make_circles(n_samples=n,noise=nz,factor=0.5,random_state=42)
    else: X,y=make_classification(n_samples=n,n_features=2,n_redundant=0,n_informative=2,n_clusters_per_class=1,flip_y=nz,random_state=42)
    X=StandardScaler().fit_transform(X); return X,y

def trn(m,X,y,ep):
    o=torch.optim.Adam(m.parameters(),lr=0.01);L=nn.CrossEntropyLoss()
    Xt,yt=torch.tensor(X,dtype=torch.float32),torch.tensor(y,dtype=torch.long)
    for _ in range(ep): o.zero_grad();l=L(m(Xt),yt);l.backward();o.step()

def atk_fgsm(m,X,y,e):
    Xt=torch.tensor(X,dtype=torch.float32,requires_grad=True);yt=torch.tensor(y,dtype=torch.long)
    nn.CrossEntropyLoss()(m(Xt),yt).backward();return(Xt+e*Xt.grad.sign()).detach().numpy()

def atk_pgd(m,X,y,e,s=10):
    Xt=torch.tensor(X,dtype=torch.float32);yt=torch.tensor(y,dtype=torch.long);Xa=Xt.clone();a=e/4
    for _ in range(s):
        Xa.requires_grad_(True);nn.CrossEntropyLoss()(m(Xa),yt).backward()
        with torch.no_grad(): Xa=Xt+torch.clamp(Xa+a*Xa.grad.sign()-Xt,-e,e)
    return Xa.detach().numpy()

def atk_noise(X,e): return X+np.random.uniform(-e,e,size=X.shape)

def mkgrid(m,X,r=160):
    xs=np.linspace(X[:,0].min()-1,X[:,0].max()+1,r);ys=np.linspace(X[:,1].min()-1,X[:,1].max()+1,r)
    xx,yy=np.meshgrid(xs,ys)
    with torch.no_grad(): Z=m(torch.tensor(np.c_[xx.ravel(),yy.ravel()],dtype=torch.float32)).argmax(1).numpy().reshape(xx.shape)
    return xx,yy,Z

def acc(m,X,y):
    with torch.no_grad(): return(m(torch.tensor(X,dtype=torch.float32)).argmax(1).numpy()==y).mean()*100

GRN="#4ADE80";RED="#F87171";GRN_B="rgba(74,222,128,0.08)";RED_B="rgba(248,113,113,0.08)"

def plot(X,y,model=None,h=420,arrows_from=None):
    fig=go.Figure()
    if model is not None:
        xx,yy,Z=mkgrid(model,X)
        fig.add_trace(go.Contour(
            x=np.linspace(xx.min(),xx.max(),Z.shape[1]),
            y=np.linspace(yy.min(),yy.max(),Z.shape[0]),
            z=Z,showscale=False,colorscale=[[0,GRN_B],[1,RED_B]],
            contours=dict(showlines=True,coloring="fill"),
            line=dict(width=1,color="rgba(255,255,255,0.08)"),hoverinfo="skip"))
    cs=[GRN if yi==0 else RED for yi in y]
    fig.add_trace(go.Scatter(x=X[:,0],y=X[:,1],mode="markers",
        marker=dict(color=cs,size=6,line=dict(width=0.4,color="rgba(0,0,0,0.3)"),opacity=0.9),
        hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>"))
    if arrows_from is not None:
        rng=np.random.RandomState(0);idx=rng.choice(len(X),size=min(30,len(X)),replace=False)
        for i in idx:
            fig.add_annotation(x=X[i,0],y=X[i,1],ax=arrows_from[i,0],ay=arrows_from[i,1],
                showarrow=True,arrowhead=3,arrowsize=0.7,arrowwidth=0.5,arrowcolor="rgba(255,255,255,0.13)")
    fig.update_layout(height=h,showlegend=False,margin=dict(t=8,b=8,l=8,r=8),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="Inter",size=12,color="#71717A"))
    fig.update_xaxes(showgrid=False,zeroline=False,showticklabels=False)
    fig.update_yaxes(showgrid=False,zeroline=False,showticklabels=False)
    return fig

# ── step indicator ──────────────────────────────────────────────────────────

def dots(cur):
    labels=["Scenario","AI learns","Attack","Outcome"]
    h=""
    for i,l in enumerate(labels):
        c="dot-done" if i<cur else ("dot-on" if i==cur else "dot-off")
        h+=f'<div class="dot {c}">{i+1}</div>'
        if i<3: h+='<div class="sline"></div>'
    h+=f'<span class="slabel">{labels[cur]}</span>'
    st.markdown(f'<div class="steps">{h}</div>',unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# STEP 0 — scenario
# ════════════════════════════════════════════════════════════════════════════

if st.session_state.step==0:
    dots(0)

    st.markdown("## Can AI systems be tricked?")

    st.markdown("""
<div class="context">
Think of an AI that decides whether a financial transaction is <em>legitimate</em>
or <em>fraudulent</em>. It looks at patterns in the data and learns a rule to
tell them apart — the same way a doctor learns to read an X-ray, except the AI
does it statistically.
<br><br>
We're going to build one of these systems right now. It's a simplified version,
but the vulnerability it exposes is <strong>identical to what exists in
production AI systems</strong> used for healthcare, finance, and national security.
<br><br>
Below you'll see the "cases" the AI will learn from. Each dot is one case.
<em>Green dots</em> are legitimate. <em>Red dots</em> are fraudulent. The AI's
job is to learn where the line is between them.
</div>
""", unsafe_allow_html=True)

    ds = st.selectbox("Pick a pattern for the cases", ["Moons","Circles","Blobs"],
                      help="These are different ways the legitimate and fraudulent cases can be distributed. All three are common in real data.")
    nz = st.slider("How much overlap between the groups?", 0.05, 0.40, 0.18, 0.05,
                    help="Real-world data is never perfectly separated. More overlap = harder problem for the AI.")

    X,y = mkdata(ds, 400, nz)

    st.markdown(f'<div class="legend"><span><span class="ldot" style="background:{GRN};"></span>Legitimate</span><span><span class="ldot" style="background:{RED};"></span>Fraudulent</span></div>', unsafe_allow_html=True)
    st.plotly_chart(plot(X,y,h=380), use_container_width=True, key="s0")
    st.caption("Each dot is one case. In a real system, these would be transactions, medical scans, or applications — anything an AI classifies.")

    if st.button("Next — let the AI learn →", type="primary"):
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=42)
        st.session_state.data = dict(X=X,y=y,Xtr=Xtr,Xte=Xte,ytr=ytr,yte=yte,ds=ds,nz=nz)
        st.session_state.step = 1
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — train
# ════════════════════════════════════════════════════════════════════════════

elif st.session_state.step==1:
    dots(1)
    d=st.session_state.data

    st.markdown("## The AI is learning to tell the groups apart.")

    st.markdown("""
<div class="context">
The AI is now studying the cases you generated. It's trying to draw a
<strong>dividing line</strong> — everything on one side it will call
<em>legitimate</em>, everything on the other side <em>fraudulent</em>.
<br><br>
This line is called the <strong>decision boundary</strong>. It's the actual
rule the AI uses to make every future decision. If this line is in the wrong
place, people get wrongly flagged — or fraud gets through.
<br><br>
You can adjust how powerful the AI is. A more complex AI can draw more
intricate lines, but it also risks "memorising" the training data instead
of learning general patterns.
</div>
""", unsafe_allow_html=True)

    hid = st.slider("AI complexity", 4, 64, 16, 4,
                     help="How many internal processing units the AI has. More = more complex decision rules.")
    ep = st.slider("Learning time", 50, 500, 200, 50,
                    help="How many passes the AI makes over the training data.")

    if st.button("Train the AI →", type="primary"):
        model=Net(hid)
        with st.spinner("The AI is studying the cases…"): trn(model,d["Xtr"],d["ytr"],ep)
        model.eval()
        cs=acc(model,d["Xte"],d["yte"])
        st.session_state.model=model; d["clean_acc"]=cs; d["hid"]=hid
        st.session_state.step=2; st.rerun()

    st.markdown(f'<div class="legend"><span><span class="ldot" style="background:{GRN};"></span>Legitimate</span><span><span class="ldot" style="background:{RED};"></span>Fraudulent</span></div>', unsafe_allow_html=True)
    st.plotly_chart(plot(d["X"],d["y"],h=380), use_container_width=True, key="s1")

# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — attack
# ════════════════════════════════════════════════════════════════════════════

elif st.session_state.step==2:
    dots(2)
    d=st.session_state.data; model=st.session_state.model

    st.markdown("## The AI is ready. Now try to fool it.")

    st.markdown(f"""
<div class="context">
The AI has learned its rule and is scoring <strong>{d['clean_acc']:.0f}%</strong>
on cases it has never seen before. The shaded regions below show its decision
zones — <em>green zone</em> = it will say "legitimate", <em>red zone</em> =
it will say "fraudulent".
<br><br>
Now imagine you're an attacker. You can make <strong>tiny, invisible
modifications</strong> to the input data — like changing a few pixels in a
medical scan, or slightly tweaking the numbers on a transaction. Not enough
for a human to notice. But enough to push the case across the AI's dividing
line, flipping its decision.
<br><br>
That's what an <strong>adversarial attack</strong> does. Choose how you'd
like to attack, and how aggressively.
</div>
""", unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1:
        atk_name=st.selectbox("Attack method",[
            "FGSM — fast, single-step",
            "PGD — stronger, multi-step (industry standard)",
            "Random noise — for comparison (not targeted)"],
            help="FGSM and PGD are real attack algorithms used by security researchers. Random noise is a baseline to show the difference.")
    with c2:
        eps=st.slider("How aggressive?", 0.0, 2.0, 0.30, 0.05,
            help="How far each data point can be moved. Think of it as the attacker's budget. Even 0.3 can be devastating.")

    if st.button("Launch the attack →", type="primary"):
        with st.spinner("Attacking…"):
            if "FGSM" in atk_name: Xa=atk_fgsm(model,d["Xte"],d["yte"],eps); al="FGSM"
            elif "PGD" in atk_name: Xa=atk_pgd(model,d["Xte"],d["yte"],eps); al="PGD"
            else: Xa=atk_noise(d["Xte"],eps); al="Random noise"
        av=acc(model,Xa,d["yte"])
        d.update(Xa=Xa,adv_acc=av,atk=al,eps=eps)
        st.session_state.step=3; st.rerun()

    st.markdown(f'<div class="legend"><span><span class="ldot" style="background:{GRN};"></span>Legitimate</span><span><span class="ldot" style="background:{RED};"></span>Fraudulent</span><span style="margin-left:4px;color:var(--muted);">Shaded = AI\'s decision zones · White line = boundary</span></div>', unsafe_allow_html=True)
    st.plotly_chart(plot(d["Xte"],d["yte"],model=model,h=420), use_container_width=True, key="s2")

# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — result
# ════════════════════════════════════════════════════════════════════════════

elif st.session_state.step==3:
    dots(3)
    d=st.session_state.data; model=st.session_state.model
    dr=d["clean_acc"]-d["adv_acc"]

    st.markdown("## Here's what the attack did.")

    m1,m2,m3=st.columns(3)
    m1.metric("Before attack", f"{d['clean_acc']:.0f}%")
    m2.metric("After attack", f"{d['adv_acc']:.0f}%")
    m3.metric("Accuracy lost", f"{dr:.0f} pts", delta=f"−{dr:.0f}", delta_color="inverse")

    st.markdown("")
    p1,p2=st.columns(2)
    with p1:
        st.caption("BEFORE — CLEAN DATA")
        st.markdown(f'<div class="legend"><span><span class="ldot" style="background:{GRN};"></span>Legit</span><span><span class="ldot" style="background:{RED};"></span>Fraud</span></div>', unsafe_allow_html=True)
        st.plotly_chart(plot(d["Xte"],d["yte"],model=model,h=400), use_container_width=True, key="r1")

    with p2:
        st.caption(f"AFTER — {d['atk'].upper()} ATTACK · ε = {d['eps']}")
        st.markdown(f'<div class="legend"><span><span class="ldot" style="background:{GRN};"></span>Legit</span><span><span class="ldot" style="background:{RED};"></span>Fraud</span><span style="color:var(--muted);margin-left:4px;">Arrows show movement</span></div>', unsafe_allow_html=True)
        st.plotly_chart(plot(d["Xa"],d["yte"],model=model,h=400,arrows_from=d["Xte"]), use_container_width=True, key="r2")

    # plain english outcome
    if dr > 15:
        msg = (f"The AI's accuracy dropped by **{dr:.0f} percentage points**. "
               f"The cases barely moved — the arrows on the right plot are tiny — but many "
               f"crossed the dividing line. In a real system, fraudulent transactions would "
               f"be approved. Legitimate ones would be flagged. And nobody looking at the "
               f"raw data would see anything wrong.")
    elif dr > 5:
        msg = (f"A **{dr:.0f}-point drop**. The attacker didn't need access to the AI's "
               f"internals. They just needed to slightly adjust the inputs. Any system making "
               f"consequential decisions at this error rate is compromised.")
    elif dr > 0.5:
        msg = (f"The overall number only moved **{dr:.1f} points** — but look at the right "
               f"plot. Individual cases crossed the line. An attacker doesn't need to fool "
               f"the whole system. They just need to flip the decision on *their* case.")
    else:
        msg = ("At this attack strength, the model held. That's useful to know — but try "
               "increasing the strength slider, or switch from random noise to FGSM or PGD.")

    st.markdown(f"""
<div class="context">
{msg}
<br><br>
This vulnerability is called <strong>adversarial brittleness</strong>. The AI
looks confident on normal data, but the rule it learned — the dividing line —
runs close to the data points. A small, calculated push is enough to cross it.
<br><br>
<strong>Most AI systems deployed in healthcare, finance, and public services
have never been tested for this.</strong> No major jurisdiction currently
requires adversarial robustness testing before deployment.
</div>
""", unsafe_allow_html=True)

    st.markdown("")
    c1,c2=st.columns(2)
    with c1:
        if st.button("← Try a different attack", use_container_width=True):
            st.session_state.step=2; st.rerun()
    with c2:
        if st.button("← Start over", use_container_width=True):
            st.session_state.step=0; st.session_state.model=None; st.session_state.data=None; st.rerun()

st.markdown(""); st.markdown("")
st.caption("Prototype · Not for redistribution")
