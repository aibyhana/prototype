"""
Interactive briefing: Why AI systems break under pressure.
Designed for decision-makers with no technical background.
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

# ── page ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Why AI Systems Break", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=DM+Sans:wght@400;500;600&display=swap');

.stApp { background: #FAFAF7; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif !important; color: #1a1a1a; }
p, li, label, .stMarkdown, .stText, span { font-family: 'DM Sans', sans-serif !important; color: #2c2c2c; }

section[data-testid="stSidebar"] { background: #1B2838; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] .stMarkdown { color: #E8E6E1 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: #B0ADA6 !important; font-size: 0.82rem;
    text-transform: uppercase; letter-spacing: 0.05em;
}

div[data-testid="stMetric"] {
    background: white; border: 1px solid #E5E2DB;
    border-radius: 6px; padding: 16px 20px;
}
div[data-testid="stMetric"] label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: #888 !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-family: 'Source Serif 4', serif !important; font-size: 1.9rem; color: #1a1a1a !important; }

.stButton > button[kind="primary"] {
    background: #8B2500; border: none; color: white;
    font-family: 'DM Sans', sans-serif; font-weight: 600;
    letter-spacing: 0.03em; border-radius: 4px;
}
.stButton > button[kind="primary"]:hover { background: #A63000; }

.streamlit-expanderHeader { font-family: 'Source Serif 4', serif !important; font-weight: 600; }
#MainMenu, footer { visibility: hidden; }

.callout {
    background: #F0EDE6; border-left: 3px solid #8B2500;
    padding: 20px 24px; border-radius: 0 6px 6px 0;
    margin: 12px 0 24px 0; font-family: 'DM Sans', sans-serif;
    color: #2c2c2c; line-height: 1.65; font-size: 0.95rem;
}
.callout-neutral {
    background: #F0EDE6; border-left: 3px solid #1B2838;
    padding: 20px 24px; border-radius: 0 6px 6px 0;
    margin: 12px 0 24px 0; font-family: 'DM Sans', sans-serif;
    color: #2c2c2c; line-height: 1.65; font-size: 0.95rem;
}
.badge {
    display: inline-block; background: #1B2838; color: #E8E6E1;
    font-family: 'DM Sans', sans-serif; font-weight: 600;
    font-size: 0.72rem; letter-spacing: 0.08em;
    text-transform: uppercase; padding: 4px 12px;
    border-radius: 3px; margin-bottom: 8px;
}
.rule { border: none; border-top: 1px solid #DDD9D0; margin: 36px 0 28px 0; }
</style>
""", unsafe_allow_html=True)

# ── ML code (hidden from the user) ─────────────────────────────────────────

class Net(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 2))
    def forward(self, x): return self.net(x)

def gen(name, n, nz):
    if name == "Two Moons":
        X, y = make_moons(n_samples=n, noise=nz, random_state=42)
    elif name == "Concentric Circles":
        X, y = make_circles(n_samples=n, noise=nz, factor=0.5, random_state=42)
    else:
        X, y = make_classification(n_samples=n, n_features=2, n_redundant=0,
            n_informative=2, n_clusters_per_class=1, flip_y=nz, random_state=42)
    sc = StandardScaler(); X = sc.fit_transform(X)
    return X, y, sc

def do_train(m, X, y, ep, lr):
    o = torch.optim.Adam(m.parameters(), lr=lr)
    L = nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    h = []
    for _ in range(ep):
        o.zero_grad(); l = L(m(Xt), yt); l.backward(); o.step(); h.append(l.item())
    return h

def do_fgsm(m, X, y, e):
    Xt = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    yt = torch.tensor(y, dtype=torch.long)
    nn.CrossEntropyLoss()(m(Xt), yt).backward()
    return (Xt + e * Xt.grad.sign()).detach().numpy()

def do_pgd(m, X, y, e, a, s):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    Xa = Xt.clone()
    for _ in range(s):
        Xa.requires_grad_(True)
        nn.CrossEntropyLoss()(m(Xa), yt).backward()
        with torch.no_grad():
            Xa = Xt + torch.clamp(Xa + a * Xa.grad.sign() - Xt, -e, e)
    return Xa.detach().numpy()

def do_noise(X, e): return X + np.random.uniform(-e, e, size=X.shape)

def grid(m, X, r=180):
    xs = np.linspace(X[:,0].min()-1, X[:,0].max()+1, r)
    ys = np.linspace(X[:,1].min()-1, X[:,1].max()+1, r)
    xx, yy = np.meshgrid(xs, ys)
    with torch.no_grad():
        Z = m(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)).argmax(1).numpy().reshape(xx.shape)
    return xx, yy, Z

def score(m, X, y):
    with torch.no_grad():
        return (m(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy() == y).mean() * 100

CA, CB = "#2D6A4F", "#9B2226"
CA_BG, CB_BG = "rgba(45,106,79,0.12)", "rgba(155,34,38,0.12)"

# ── intro ───────────────────────────────────────────────────────────────────

st.markdown("""
<div style="font-family:'DM Sans',sans-serif; font-size:0.78rem; font-weight:600;
            letter-spacing:0.1em; text-transform:uppercase; color:#8B2500; margin-bottom:4px;">
    Interactive Briefing · Prototype
</div>
""", unsafe_allow_html=True)

st.markdown("## Why AI systems break under pressure")

st.markdown("""
<div class="callout">
Imagine an AI that screens medical images, flags fraudulent transactions, or
decides who gets a loan. These systems learn rules from data — and they can be
<strong>very accurate</strong>. But they share a hidden weakness: with the right
mathematical trick, an attacker can make tiny, invisible changes to the input
and cause the AI to make the <strong>wrong decision</strong>.
<br><br>
This isn't a theoretical risk. It has been demonstrated on every major AI
architecture in use today. And most systems deployed in the real world have
<strong>never been tested</strong> for it.
<br><br>
This tool lets you see it happen, hands-on, in about 30 seconds.
</div>
""", unsafe_allow_html=True)

# ── sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Settings")
    st.caption("Don't worry about getting these 'right' — the defaults work well. Change them later to explore.")

    st.markdown("---")

    st.markdown('<div class="badge">The data</div>', unsafe_allow_html=True)
    st.markdown("The AI will learn to tell apart two groups of dots. Pick a shape for those groups.")
    ds = st.selectbox("Shape", ["Two Moons", "Concentric Circles", "Gaussian Blobs"], label_visibility="collapsed")
    ns = st.slider("How many dots", 200, 1000, 400, 100)
    nz = st.slider("How messy the data is", 0.05, 0.50, 0.20, 0.05)

    st.markdown("---")

    st.markdown('<div class="badge">The AI model</div>', unsafe_allow_html=True)
    st.markdown("A small neural network. More neurons = more complex rules it can learn.")
    hid = st.slider("Neurons", 4, 64, 16, 4)
    ep = st.slider("Training rounds", 50, 500, 200, 50)

    st.markdown("---")

    st.markdown('<div class="badge">The attack</div>', unsafe_allow_html=True)
    st.markdown("How we try to break it. FGSM is fast and simple. PGD is the standard stress test used in research. Random noise is a control — no intelligence behind it.")
    atk = st.selectbox("Method", ["FGSM (fast, one step)", "PGD (stronger, many steps)", "Random noise (for comparison)"], label_visibility="collapsed")
    eps = st.slider("Attack strength", 0.0, 2.0, 0.30, 0.05,
                    help="How much each data point is allowed to move. Even small values can cause big drops.")

    ps, pa = 10, 0.05
    if "PGD" in atk:
        ps = st.slider("PGD steps", 1, 40, 10)

    st.markdown("---")
    go_btn = st.button("Run the experiment", type="primary", use_container_width=True)

# ── landing ─────────────────────────────────────────────────────────────────

if not go_btn:
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown("### How this works")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**① You generate data**")
        st.markdown(
            "Two groups of dots appear on a 2D plane — think of them as "
            "'approve' vs 'reject', or 'healthy' vs 'at risk'. The pattern "
            "isn't perfectly clean, just like real-world data."
        )
    with c2:
        st.markdown("**② The AI learns a rule**")
        st.markdown(
            "A neural network trains on the data and draws a line (the "
            "*decision boundary*) separating the two groups. It gets tested "
            "on data it hasn't seen before, and you'll see its accuracy."
        )
    with c3:
        st.markdown("**③ We attack it**")
        st.markdown(
            "We nudge each test point by a tiny, calculated amount — small "
            "enough that a human wouldn't notice — and measure how many "
            "predictions flip. The gap between the before and after "
            "accuracy is the vulnerability."
        )

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    with st.expander("Glossary — what do these terms mean?"):
        st.markdown("""
**Decision boundary** — the dividing line the AI learned. Everything on one
side gets labelled "Group A", everything on the other side "Group B". It's the
rule the system is actually using.

**Adversarial attack** — a way to fool an AI by making tiny, deliberate changes
to its input. Unlike random errors, these are mathematically crafted to cause
the maximum damage with the minimum change. Think of it as finding the exact
angle to tap a glass so it shatters.

**FGSM** (Fast Gradient Sign Method) — the simplest adversarial attack. It
looks at which direction would increase the AI's error the most, and nudges
the data one step in that direction.

**PGD** (Projected Gradient Descent) — a stronger version that takes many
small steps instead of one big one. This is the standard stress-test used
by researchers to evaluate whether an AI is robust.

**ε (epsilon)** — the budget the attacker has. It controls how far each data
point is allowed to move. A smaller ε means a subtler, harder-to-detect attack.
Even very small values can break poorly designed systems.

**Accuracy** — the percentage of predictions the AI gets right. If a system
goes from 95% to 60% after an attack, that 35-point drop is the vulnerability
you'd want regulation to address.
        """)

    st.markdown("""
<div class="callout-neutral">
<strong>Ready?</strong> Use the settings on the left (the defaults are fine for
a first run) and press <strong>Run the experiment</strong>.
</div>
    """, unsafe_allow_html=True)

    st.stop()

# ── run ─────────────────────────────────────────────────────────────────────

X, y, sc = gen(ds, ns, nz)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

model = Net(hid)
with st.spinner("The AI is learning from the data…"):
    hist = do_train(model, Xtr, ytr, ep, 0.01)
model.eval()

cs = score(model, Xte, yte)

with st.spinner("Running the attack…"):
    if "FGSM" in atk:
        Xa = do_fgsm(model, Xte, yte, eps); al = "FGSM"
    elif "PGD" in atk:
        Xa = do_pgd(model, Xte, yte, eps, pa, ps); al = "PGD"
    else:
        Xa = do_noise(Xte, eps); al = "Random noise"

av = score(model, Xa, yte)
dr = cs - av

# ── results ─────────────────────────────────────────────────────────────────

st.markdown('<hr class="rule">', unsafe_allow_html=True)

st.markdown("### The result")
st.markdown(
    "The AI was tested on data it had never seen. First with clean data, "
    "then with the attacked version. Same model, nearly the same data — "
    "very different outcomes."
)

m1, m2, m3 = st.columns(3)
m1.metric("Before the attack", f"{cs:.1f}%")
m2.metric("After the attack", f"{av:.1f}%")
m3.metric("Accuracy lost", f"{dr:.1f} points", delta=f"−{dr:.1f}pp", delta_color="inverse")

# ── plots ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="callout-neutral">
<strong>How to read these plots:</strong> Each dot is a data point the AI has to
classify. The <span style="color:#2D6A4F; font-weight:600;">green</span> and
<span style="color:#9B2226; font-weight:600;">red</span> shaded regions are the
AI's "decision zones" — if a dot lands in the wrong zone, the AI gets it wrong.
The thin line between zones is the <strong>decision boundary</strong>.<br><br>
On the right, the small arrows show how each point was moved by the attack.
Notice they barely moved — but many crossed the line.
</div>
""", unsafe_allow_html=True)

xx, yy, Z = grid(model, X)
cp = [CA if yi == 0 else CB for yi in yte]

fig = make_subplots(rows=1, cols=2,
    subplot_titles=("Clean data", f"After {al} attack (strength = {eps})"),
    horizontal_spacing=0.07)

for c in [1, 2]:
    fig.add_trace(go.Contour(
        x=np.linspace(xx.min(), xx.max(), Z.shape[1]),
        y=np.linspace(yy.min(), yy.max(), Z.shape[0]),
        z=Z, showscale=False,
        colorscale=[[0, CA_BG], [1, CB_BG]],
        contours=dict(showlines=True, coloring="fill"),
        line=dict(width=1.5, color="rgba(0,0,0,0.18)"),
        hoverinfo="skip"), row=1, col=c)

fig.add_trace(go.Scatter(x=Xte[:,0], y=Xte[:,1], mode="markers",
    marker=dict(color=cp, size=5.5, line=dict(width=0.4, color="white")),
    hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>"), row=1, col=1)

fig.add_trace(go.Scatter(x=Xa[:,0], y=Xa[:,1], mode="markers",
    marker=dict(color=cp, size=5.5, line=dict(width=0.4, color="white")),
    hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>"), row=1, col=2)

rng = np.random.RandomState(0)
idx = rng.choice(len(Xte), size=min(35, len(Xte)), replace=False)
for i in idx:
    fig.add_annotation(x=Xa[i,0], y=Xa[i,1], ax=Xte[i,0], ay=Xte[i,1],
        xref="x2", yref="y2", axref="x2", ayref="y2",
        showarrow=True, arrowhead=3, arrowsize=0.8, arrowwidth=0.7,
        arrowcolor="rgba(0,0,0,0.20)")

fig.update_layout(height=480, showlegend=False,
    margin=dict(t=36, b=16, l=16, r=16),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", size=12))
fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
fig.update_annotations(font_size=13)

st.plotly_chart(fig, use_container_width=True)

# ── so what ─────────────────────────────────────────────────────────────────

st.markdown('<hr class="rule">', unsafe_allow_html=True)
st.markdown("### So what does this mean?")

if dr > 15:
    v = (
        f"The attack reduced the AI's accuracy by <strong>{dr:.0f} percentage "
        f"points</strong>. That's a catastrophic failure. If this were a medical "
        f"screening system, readings could flip from 'healthy' to 'at risk'. If "
        f"it were a fraud detection system, fraudulent transactions could sail "
        f"through. And the input data barely changed — a human looking at the "
        f"'before' and 'after' would see no difference."
    )
elif dr > 5:
    v = (
        f"The AI lost <strong>{dr:.0f} percentage points</strong> of accuracy. "
        f"In any system where decisions have real consequences — lending, hiring, "
        f"diagnostics — that kind of degradation is unacceptable. And the attack "
        f"required no insider access to the AI. Just the ability to slightly "
        f"modify inputs."
    )
elif dr > 0.5:
    v = (
        f"The overall accuracy only dropped by <strong>{dr:.1f} points</strong>, "
        f"but look at the plot — individual points did cross the boundary. In "
        f"practice, an attacker doesn't need to fool the whole system. They only "
        f"need to fool it on <em>their</em> specific input."
    )
else:
    v = (
        "At this attack strength, the model held up. That's worth noting — not "
        "every attack succeeds. But try increasing the strength slider, or "
        "switch from random noise to FGSM or PGD, and you'll see when it "
        "starts to break."
    )

st.markdown(f"""
<div class="callout">
{v}
<br><br>
What you're seeing is a property called <strong>adversarial brittleness</strong>.
The AI's high accuracy on normal data masks a hidden fragility: the rules it
learned are easy to exploit if you know where to push. The 'decision boundary'
— the dividing line between its yes and no — often runs dangerously close to
the data points.
<br><br>
Most AI systems used in high-stakes settings today have never been tested this way.
There is currently no legal requirement to do so in any major jurisdiction.
</div>
""", unsafe_allow_html=True)

st.markdown("### What to try next")
st.markdown("""
- **Increase attack strength** (the ε slider) and watch accuracy collapse.
  At what point would the failure be unacceptable in a system you regulate?
- **Switch attack method** from random noise to FGSM or PGD. Notice the
  difference — a calculated attack is far more damaging than random error.
- **Reduce model complexity** (fewer neurons). Simpler models sometimes resist
  attacks better — but also make more mistakes on clean data. This is the
  robustness-accuracy tradeoff that engineers deal with.
- **Change the data shape** to concentric circles. Some patterns are harder
  to learn — and harder to attack.
""")

# ── technical (hidden) ──────────────────────────────────────────────────────

with st.expander("For technical staff — training details"):
    st.markdown(f"""
    Model: 2-layer ReLU network, {hid} hidden units per layer.
    Trained for {ep} rounds with Adam (lr=0.01) on {len(Xtr)} samples.
    Final training loss: {hist[-1]:.4f}.
    Attack: {al}, ε={eps}{f', {ps} steps, α={pa}' if 'PGD' in atk else ''}.
    """)
    lfig = go.Figure()
    lfig.add_trace(go.Scatter(y=hist, mode="lines", line=dict(color="#1B2838", width=1.8)))
    lfig.update_layout(xaxis_title="Round", yaxis_title="Loss", height=240,
        margin=dict(t=8, b=36, l=44, r=16),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", size=11))
    lfig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    lfig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    st.plotly_chart(lfig, use_container_width=True)

# ── footer ──────────────────────────────────────────────────────────────────

st.markdown('<hr class="rule">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding: 8px 0 20px 0;">
    <span style="font-family:'DM Sans',sans-serif; font-size:0.78rem; color:#999; letter-spacing:0.04em;">
        Prototype · Not for redistribution
    </span>
</div>
""", unsafe_allow_html=True)
