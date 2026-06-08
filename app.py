import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import textwrap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def outliers_to_nan(df):
    df = df.copy()
    df.loc[(df["u"] >= 0) & (df["u"] < 2.5), "u"] = np.nan
    df.loc[df["i"] >= 27, "i"] = np.nan
    return df


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        new_cols = ['u-g', 'g-r', 'r-i', 'i-z']
        X = X.drop(columns=[c for c in new_cols if c in X.columns])
        X['u-g'] = X['u'] - X['g']
        X['g-r'] = X['g'] - X['r']
        X['r-i'] = X['r'] - X['i']
        X['i-z'] = X['i'] - X['z']
        return X


st.set_page_config(
    page_title="Stellar Classifier",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
  background-color: #060e20;
  color: #c8daf4;
  font-family: 'Inter', sans-serif;
}

/* ── BACKGROUND ── */
.stApp {
  background:
    radial-gradient(ellipse at 30% 0%,  rgba(30, 60, 160, 0.22) 0%, transparent 50%),
    radial-gradient(ellipse at 75% 20%, rgba(20, 45, 120, 0.15) 0%, transparent 45%),
    radial-gradient(ellipse at 50% 70%, rgba(10, 30,  90, 0.10) 0%, transparent 50%),
    #060e20;
}

/* ── HERO SECTION ── */
.hero-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 5rem 2rem 3rem;
  width: 100%;
}
.hero-eyebrow {
  font-family: 'Orbitron', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.3em;
  color: #6a8ec8;
  text-transform: uppercase;
  margin-bottom: 1.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
}
.hero-eyebrow::before, .hero-eyebrow::after {
  content: '';
  width: 40px;
  height: 1px;
  background: linear-gradient(90deg, transparent, #4a70b0);
}
.hero-title {
  font-family: 'Orbitron', monospace;
  font-size: clamp(2.2rem, 4vw, 3.4rem);
  font-weight: 700;
  color: #e8f2ff;
  letter-spacing: 0.02em;
  line-height: 1.15;
  margin: 0 0 1.5rem;
}
.hero-title span { color: #80b4ff; }
.hero-desc {
  color: #96b4d4;
  font-size: 1rem;
  line-height: 1.85;
  margin: 0 0 2rem;
  max-width: 540px;
  text-align: center;
}
.hero-stats {
  display: inline-flex;
  gap: 0;
  border: 1px solid rgba(80, 125, 210, 0.45);
  border-radius: 14px;
  overflow: hidden;
  margin-bottom: 2.5rem;
}
.stat-item {
  padding: 0.9rem 1.8rem;
  border-right: 1px solid rgba(80, 125, 210, 0.3);
  font-size: 0.82rem;
  color: #7a9cc0;
  background: rgba(10, 22, 58, 0.7);
}
.stat-item:last-child { border-right: none; }

/* ── CLASS GRID ── */
.class-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 0.75rem;
  margin: 0 0 2rem;
  width: 100%;
  max-width: 660px;
}
.class-card {
  background: rgba(10, 22, 55, 0.7);
  border: 1px solid rgba(70, 110, 185, 0.32);
  border-radius: 14px;
  padding: 1.5rem 1rem;
  text-align: center;
  transition: border-color 0.2s, background 0.2s;
}
.class-card:hover {
  border-color: rgba(100, 150, 240, 0.5);
  background: rgba(15, 30, 75, 0.8);
}
.class-emoji { font-size: 2.2rem; margin-bottom: 0.6rem; }
.class-name  {
  font-family: 'Orbitron', monospace;
  font-size: 0.67rem; letter-spacing: 0.18em;
  color: #6090d0; margin-bottom: 0.5rem;
}
.class-desc  { font-size: 0.8rem; color: #5070a0; line-height: 1.6; }
.stat-item b { display: block; font-size: 1.2rem; color: #a8d0ff; font-weight: 700; margin-bottom: 0.15rem; }

/* ── SECTION DIVIDER ── */
.section-sep {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin: 1rem 0 1.5rem;
  color: #5a78a8;
  font-family: 'Orbitron', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
}
.section-sep::before, .section-sep::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(80, 125, 200, 0.5), transparent);
}

/* ── SKY MAP ── */
.map-wrap {
  background: rgba(4, 9, 22, 0.95);
  border: 1px solid rgba(70, 110, 185, 0.38);
  border-radius: 16px;
  padding: 1.1rem 1.1rem 0.4rem;
  margin-bottom: 0.6rem;
}
.map-footer { font-size: 0.64rem; color: #3d5a80; text-align: center; padding-bottom: 0.5rem; }

/* ── PANEL CARD ── */
.panel {
  background: rgba(8, 20, 50, 0.85);
  border: 1px solid rgba(65, 105, 180, 0.35);
  border-radius: 14px;
  padding: 1.2rem 1.3rem;
  margin-bottom: 0.8rem;
}
.panel-title {
  font-family: 'Orbitron', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.22em;
  color: #5a7aaa;
  text-transform: uppercase;
  margin-bottom: 0.8rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.panel-title::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, rgba(70, 115, 195, 0.4), transparent);
}

/* ── REDSHIFT SECTION LABEL (no card) ── */
.sec-lbl-inline {
  font-family: 'Orbitron', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.22em;
  color: #5a7aaa;
  text-transform: uppercase;
  margin-bottom: 0.2rem;
}

/* ── INPUT LABELS ── */
.inp-lbl {
  font-size: 0.75rem;
  color: #8aaace;
  letter-spacing: 0.03em;
  margin-bottom: 0.1rem;
  font-weight: 500;
}

/* ── INPUTS ── */
div[data-testid="stNumberInput"] input {
  background: rgba(8, 16, 40, 0.9) !important;
  border: 1px solid rgba(75, 115, 195, 0.5) !important;
  border-radius: 8px !important;
  color: #b0cce8 !important;
  font-size: 0.86rem !important;
}
div[data-testid="stNumberInput"] input:focus {
  border-color: rgba(100, 155, 245, 0.75) !important;
  color: #d0e4ff !important;
  box-shadow: 0 0 0 3px rgba(80, 135, 230, 0.15) !important;
  outline: none !important;
}
div[data-testid="stSelectbox"] > div > div {
  background: rgba(8, 16, 40, 0.9) !important;
  border: 1px solid rgba(75, 115, 195, 0.5) !important;
  border-radius: 8px !important;
  color: #b0cce8 !important;
  font-size: 0.86rem !important;
}

/* ── BUTTONS: solid colours so they're always visible ── */
.stButton > button {
  background: #2758b8;
  color: #ddeeff;
  border: 1px solid #5090e0;
  border-radius: 8px;
  font-family: 'Inter', sans-serif;
  font-size: 0.85rem;
  font-weight: 600;
  transition: all 0.18s;
  padding: 0.5rem 0.9rem;
  letter-spacing: 0.01em;
}
.stButton > button:hover {
  background: #3570e0;
  color: #eef6ff;
  border-color: #78b0ff;
  box-shadow: 0 5px 20px rgba(50, 110, 240, 0.5);
  transform: translateY(-1px);
}

/* ── RZ BAR ── */
.rz-wrap { margin: 0.2rem 0 0.5rem; }
.rz-bar {
  height: 4px; border-radius: 2px;
  background: linear-gradient(90deg,
    #5590e0 0%, #5590e0 1.43%,
    #3870c0 1.43%, #3870c0 21.4%,
    #6048a8 21.4%, #6048a8 100%);
  position: relative; margin-bottom: 5px;
}
.rz-cur {
  position: absolute; top: 50%;
  transform: translate(-50%,-50%);
  width: 10px; height: 10px;
  background: #78aaf8; border-radius: 50%;
  box-shadow: 0 0 8px rgba(120,170,248,0.8);
}
.rz-lbs {
  display: flex; justify-content: space-between;
  font-size: 0.64rem; color: #3a5878;
}

/* ── DIVIDER LINE ── */
.div-line {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(60,100,175,0.3), transparent);
  margin: 0.85rem 0;
}

/* ── RESULT ── */
.result-wrap { animation: fadeUp 0.4s cubic-bezier(0.22,0.9,0.36,1); }
@keyframes fadeUp {
  from { opacity:0; transform:translateY(16px); }
  to   { opacity:1; transform:none; }
}
.result-box {
  background: rgba(8, 18, 48, 0.9);
  border: 1px solid rgba(65, 105, 190, 0.45);
  border-radius: 14px;
  padding: 2rem 1.5rem 1.75rem;
  text-align: center;
  position: relative; overflow: hidden;
}
.result-box::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(circle at 50% 20%, rgba(70,130,230,0.1) 0%, transparent 65%);
  pointer-events: none;
}
.result-icon { font-size: 3.2rem; display:block; margin-bottom:0.5rem; }
.result-lbl  {
  font-family: 'Orbitron', monospace;
  font-size: 1.8rem; font-weight: 700;
  color: #c0d8ff; letter-spacing: 0.06em;
  margin: 0.15rem 0 0.6rem;
}
.result-desc { color: #6888a8; font-size: 0.86rem; line-height: 1.7; }
.result-conf {
  display: inline-block;
  border: 1px solid rgba(80,125,210,0.45);
  border-radius: 100px;
  padding: 0.22rem 0.8rem;
  font-size: 0.78rem; color: #7090b8;
  margin-top: 0.8rem;
}
.result-conf b { color: #a8d0ff; }

/* ── PROB BARS ── */
.prob-row  { display:flex; align-items:center; gap:0.6rem; margin-bottom:0.48rem; }
.prob-lbl  { width:72px; font-size:0.8rem; color:#5a7898; text-align:right; flex-shrink:0; }
.prob-bg   { flex:1; height:7px; background:rgba(255,255,255,0.06); border-radius:4px; overflow:hidden; }
.prob-fill { height:100%; border-radius:4px; background:rgba(75,120,210,0.55); }
.prob-fill.top { background: linear-gradient(90deg, #5088e8, #90c0ff); }
.prob-pct  { width:36px; font-size:0.8rem; color:#5a7898; text-align:right; flex-shrink:0; }
.top-lbl   { color:#c0daff !important; font-weight:600; }

/* ── COLOR INDEX ── */
.ci-grid { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:0.45rem; }
.ci-chip {
  background:rgba(8,20,52,0.9);
  border:1px solid rgba(70,110,190,0.38);
  border-radius:8px; padding:0.5rem 0.3rem;
  text-align:center; font-size:0.73rem; color:#5a7898;
}
.ci-v { font-size:0.95rem; font-weight:600; display:block; color:#a0c4f8; margin-bottom:0.1rem; }

/* ── AWAIT ── */
.await-box {
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  gap:0.8rem; text-align:center;
  padding:4rem 1rem; min-height:340px;
  border:1px solid rgba(45,80,145,0.25); border-radius:14px;
}
.await-ic { animation:awp 3s ease-in-out infinite; font-size:2.5rem; }
@keyframes awp { 0%,100%{opacity:0.15;} 50%{opacity:0.45;} }
.await-h { font-family:'Orbitron',monospace; font-size:0.62rem; letter-spacing:0.22em; color:#2d4870; }
.await-s { font-size:0.78rem; color:#2d4870; line-height:1.65; max-width:200px; }

/* ── APP SCREEN HEADER ── */
.app-header { padding: 1.5rem 0 0.5rem; }
.app-header-title {
  font-family: 'Orbitron', monospace;
  font-size: 1.2rem; font-weight: 600;
  color: #d0e4ff; letter-spacing: 0.04em;
  margin-bottom: 0.3rem;
}
.app-header-sub { font-size: 0.84rem; color: #5a7898; }

/* ── BACK BUTTON ── */
div[data-testid="stHorizontalBlock"] .stButton:first-child > button {
  background: transparent !important;
  color: #4a6888 !important;
  border-color: rgba(50,85,150,0.2) !important;
  padding: 0.3rem 0.65rem !important;
  font-size: 0.76rem !important;
}
div[data-testid="stHorizontalBlock"] .stButton:first-child > button:hover {
  color: #90b8e8 !important;
  border-color: rgba(80,130,220,0.4) !important;
  box-shadow: none !important;
}

/* ══ METHODOLOGY SECTION ══════════════════════════════════════════════════ */
.meth-wrap {
  width: 100%;
  padding: 3rem 0 4rem;
  max-width: 820px;
  margin: 0 auto;
}
.meth-title {
  font-family: 'Orbitron', monospace;
  font-size: 1.5rem; font-weight: 700;
  color: #d8eaff; text-align: center;
  margin-bottom: 2rem;
}
.meth-eyebrow {
  font-size: 0.62rem; letter-spacing: 0.28em;
  color: #4a6a9a; text-transform: uppercase;
  margin-bottom: 0.5rem;
}
.meth-card {
  background: rgba(8, 20, 52, 0.85);
  border: 1px solid rgba(65, 105, 185, 0.3);
  border-radius: 16px; padding: 1.4rem 1.6rem;
}
.mc-hdr {
  font-family: 'Orbitron', monospace;
  font-size: 0.62rem; letter-spacing: 0.2em;
  color: #5a7aaa; text-transform: uppercase;
  margin-bottom: 1.2rem;
  display: flex; align-items: center; gap: 0.5rem;
}
.mc-hdr::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, rgba(70,110,195,0.3), transparent);
}

/* Pipeline */
.pipe-flow { display: flex; flex-direction: column; gap: 0; }
.pipe-step {
  display: flex; gap: 1rem; align-items: flex-start;
  background: rgba(255,255,255,0.025);
  border: 1px solid rgba(55,90,165,0.22);
  border-radius: 12px; padding: 0.9rem 1rem;
}
.pipe-accent { border-color: rgba(80,130,220,0.45); background: rgba(40,80,190,0.1); }
.pipe-ensemble-step { border-color: rgba(90,140,230,0.5); background: rgba(30,65,170,0.12); }
.pipe-output { border-color: rgba(60,200,120,0.3); background: rgba(30,120,70,0.08); }
.pipe-ico { font-size: 1.5rem; flex-shrink: 0; margin-top: 0.1rem; }
.pipe-body { flex: 1; }
.pipe-name {
  font-size: 0.9rem; font-weight: 600; color: #c8deff;
  margin-bottom: 0.3rem; display: flex; align-items: center; gap: 0.5rem;
}
.pipe-badge {
  font-size: 0.62rem; background: rgba(80,130,230,0.25);
  border: 1px solid rgba(80,130,230,0.45); border-radius: 100px;
  padding: 0.1rem 0.5rem; color: #78aaee; font-weight: 400;
  letter-spacing: 0.05em;
}
.pipe-desc { font-size: 0.78rem; color: #4a6888; line-height: 1.55; margin-bottom: 0.4rem; }
.pipe-tags { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-top: 0.3rem; }
.ptag {
  background: rgba(55,90,175,0.2); border: 1px solid rgba(70,110,200,0.35);
  border-radius: 6px; padding: 0.15rem 0.5rem;
  font-size: 0.74rem; color: #7aabee;
}
.ptag-g { background:rgba(62,130,220,0.15); border-color:rgba(62,130,220,0.35); color:#7ab8ff; }
.ptag-s { background:rgba(200,160,50,0.12); border-color:rgba(200,160,50,0.3);  color:#d4a84a; }
.ptag-q { background:rgba(160,100,220,0.12); border-color:rgba(160,100,220,0.3);color:#b888ee; }
.pipe-split {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 0.5rem; margin-top: 0.35rem;
}
.pipe-branch {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(50,80,150,0.2);
  border-radius: 8px; padding: 0.4rem 0.6rem;
  font-size: 0.74rem; color: #4a6888; line-height: 1.5;
}
.branch-label {
  font-size: 0.65rem; font-weight: 600;
  color: #6a9ad0; text-transform: uppercase;
  letter-spacing: 0.08em; display: block;
  margin-bottom: 0.2rem;
}
.pipe-arrow {
  text-align: center; color: #2d4870;
  font-size: 1.1rem; padding: 0.15rem 0;
  line-height: 1;
}

/* Ensemble chips */
.ensemble-chips { display: flex; gap: 0.6rem; align-items: center; margin-top: 0.4rem; flex-wrap: wrap; }
.echip {
  flex: 1; min-width: 130px;
  border-radius: 10px; padding: 0.6rem 0.8rem;
  text-align: center;
}
.echip-name { font-size: 0.82rem; font-weight: 600; margin-bottom: 0.25rem; }
.echip-sub  { font-size: 0.7rem; line-height: 1.4; opacity: 0.75; }
.echip-xgb  { background:rgba(50,120,220,0.15); border:1px solid rgba(50,120,220,0.35); color:#7ab4ff; }
.echip-lgbm { background:rgba(50,180,120,0.12); border:1px solid rgba(50,180,120,0.32); color:#60d898; }
.echip-cat  { background:rgba(200,100,50,0.12); border:1px solid rgba(200,100,50,0.3);  color:#e08860; }
.echip-plus { font-size:1.2rem; color:#2d4870; font-weight:300; }

/* Two-column layout */
.meth-two-col { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }

/* K-Fold */
.kf-title { font-size:0.88rem; font-weight:600; color:#b0c8e8; margin-bottom:0.25rem; }
.kf-desc  { font-size:0.75rem; color:#3a5270; margin-bottom:0.9rem; line-height:1.5; }
.kfold-grid { display:flex; flex-direction:column; gap:4px; margin-bottom:0.6rem; }
.kf-row { display:flex; gap:4px; }
.kf-cell { flex:1; height:20px; border-radius:4px; }
.kf-train { background:rgba(45,80,170,0.35); border:1px solid rgba(55,95,195,0.2); }
.kf-test  { background:rgba(80,160,255,0.65); border:1px solid rgba(100,180,255,0.6);
            box-shadow:0 0 6px rgba(80,160,255,0.3); }
.kf-legend { display:flex; gap:1rem; font-size:0.72rem; margin-bottom:0.75rem; }
.kf-leg-test  { color:#70c0ff; }
.kf-leg-train { color:#3d5888; }
.kf-metric {
  font-size:0.75rem; color:#3a5270; line-height:1.55;
  border-top:1px solid rgba(50,80,150,0.2); padding-top:0.7rem; margin-top:0.25rem;
}
.kf-metric b { color:#7aaae0; }

/* Score bars */
.scores-list { display:flex; flex-direction:column; gap:0.55rem; }
.score-row { display:flex; align-items:center; gap:0.6rem; }
.score-name { width:130px; font-size:0.76rem; color:#5a7898; flex-shrink:0; }
.score-bar-wrap { flex:1; height:6px; background:rgba(255,255,255,0.05); border-radius:3px; overflow:hidden; }
.score-bar { height:100%; border-radius:3px; background:rgba(55,95,195,0.55); }
.score-bar-top { background:linear-gradient(90deg, #4878d8, #78aaf8); }
.score-val { width:44px; font-size:0.76rem; color:#5a7898; text-align:right; flex-shrink:0; }
.score-winner .score-name { color:#a0c4f8; font-weight:600; }
.score-winner .score-val  { color:#a0c4f8; font-weight:600; }

/* ── HIDE STREAMLIT CHROME ── */
footer, #MainMenu, header { visibility:hidden; }
[data-testid="stDecoration"] { display:none; }
</style>
""", unsafe_allow_html=True)

# ─── MODELS & DATA ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    pipeline = joblib.load('models/model.pkl')
    le = joblib.load('models/label_encoder.pkl')
    return pipeline, le

@st.cache_data
def load_sky_sample():
    try:
        df = pd.read_csv('data/train.csv', usecols=['alpha', 'delta', 'class'])
        parts = []
        for cls_name in df['class'].unique():
            subset = df[df['class'] == cls_name]
            parts.append(subset.sample(min(1500, len(subset)), random_state=42))
        sample = pd.concat(parts, ignore_index=True)
        # Convert equatorial → galactic coordinates
        ra  = np.radians(sample['alpha'].values)
        dec = np.radians(sample['delta'].values)
        ra0, dec0 = np.radians(192.8595), np.radians(27.1284)
        l0 = np.radians(122.9320)
        b = np.arcsin(np.sin(dec0)*np.sin(dec) + np.cos(dec0)*np.cos(dec)*np.cos(ra - ra0))
        x = np.cos(dec)*np.sin(ra - ra0)
        y = np.sin(dec0)*np.cos(dec)*np.cos(ra - ra0) - np.cos(dec0)*np.sin(dec)
        l = np.degrees(l0 - np.arctan2(x, y)) % 360
        sample['gal_l'] = l
        sample['gal_b'] = np.degrees(b)
        return sample
    except Exception:
        return None

pipeline, le = load_models()
sky_df = load_sky_sample()

# ─── CONSTANTS ─────────────────────────────────────────────────────────────────
CLASS_META = {
    'GALAXY': {
        'emoji': '🌌',
        'desc': 'A gravitationally-bound system of billions of stars, gas, dust and dark matter spanning thousands of light-years.',
    },
    'STAR': {
        'emoji': '⭐',
        'desc': 'A luminous ball of plasma held by its own gravity, generating energy through nuclear fusion in its core.',
    },
    'QSO': {
        'emoji': '✨',
        'desc': 'Quasi-Stellar Object - an extremely luminous active galactic nucleus powered by a supermassive black hole.',
    },
}

PRESETS = {
    'GALAXY': {'u': 21.90, 'g': 19.90, 'r': 18.50, 'i_band': 18.00, 'z': 17.70,
               'redshift': 0.25, 'spectral_type': 'M', 'galaxy_population': 'Red_Sequence',
               'alpha': 147.7, 'delta': 16.9},
    'STAR':   {'u': 18.84, 'g': 17.99, 'r': 18.46, 'i_band': 18.23, 'z': 19.20,
               'redshift': 0.11, 'spectral_type': 'O/B', 'galaxy_population': 'Blue_Cloud',
               'alpha': 254.9, 'delta': 38.7},
    'QSO':    {'u': 21.04, 'g': 21.08, 'r': 21.17, 'i_band': 20.58, 'z': 20.56,
               'redshift': 2.82, 'spectral_type': 'O/B', 'galaxy_population': 'Blue_Cloud',
               'alpha': 179.8, 'delta': 35.3},
}

SPEC_OPTS = ['M', 'G/K', 'A/F', 'O/B']
POP_OPTS  = ['Blue_Cloud', 'Red_Sequence']

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
for _k, _v in {
    'u': 22.0, 'g': 21.0, 'r': 20.0, 'i_band': 19.0, 'z': 19.0,
    'redshift': 0.5, 'alpha': 180.0, 'delta': 30.0,
    'spectral_type': 'M', 'galaxy_population': 'Red_Sequence',
    'result': None, 'show_app': False,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def rz_bar(value: float) -> str:
    pct = min((value / 7.0) * 100, 98.5)
    return (f'<div class="rz-wrap">'
            f'<div class="rz-bar"><div class="rz-cur" style="left:{pct:.2f}%"></div></div>'
            f'<div class="rz-lbs"><span>star &lt;0.1</span><span>galaxy 0.1–1.5</span><span>qso &gt;0.5</span></div>'
            f'</div>')

def prob_bars_html(classes, proba) -> str:
    top = int(np.argmax(proba))
    html = ''
    for idx, (cls, p) in enumerate(zip(classes, proba)):
        lbl_cls  = 'top-lbl' if idx == top else ''
        fill_cls = 'top'     if idx == top else ''
        html += (f'<div class="prob-row">'
                 f'<div class="prob-lbl {lbl_cls}">{CLASS_META[cls]["emoji"]} {cls}</div>'
                 f'<div class="prob-bg"><div class="prob-fill {fill_cls}" style="width:{p*100:.1f}%"></div></div>'
                 f'<div class="prob-pct {lbl_cls}">{p*100:.1f}%</div>'
                 f'</div>')
    return html

def _to_galactic(ra_deg, dec_deg):
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    ra0, dec0 = np.radians(192.8595), np.radians(27.1284)
    l0 = np.radians(122.9320)
    b = np.degrees(np.arcsin(
        np.sin(dec0)*np.sin(dec) + np.cos(dec0)*np.cos(dec)*np.cos(ra - ra0)
    ))
    x = np.cos(dec)*np.sin(ra - ra0)
    y = np.sin(dec0)*np.cos(dec)*np.cos(ra - ra0) - np.cos(dec0)*np.sin(dec)
    l = np.degrees(l0 - np.arctan2(x, y)) % 360
    lon = float(l - 360 if l > 180 else l)
    return lon, float(b)

def make_sky_map(user_alpha, user_delta):
    MAP_CLR = {'GALAXY': '#c84848', 'STAR': '#4878d0', 'QSO': '#48b858'}
    fig = go.Figure()

    if sky_df is not None and 'gal_l' in sky_df.columns:
        for cls in ['GALAXY', 'STAR', 'QSO']:
            sub = sky_df[sky_df['class'] == cls]
            lon = np.where(sub['gal_l'].values > 180,
                           sub['gal_l'].values - 360,
                           sub['gal_l'].values)
            fig.add_trace(go.Scattergeo(
                lon=lon, lat=sub['gal_b'].values,
                mode='markers',
                name=f"{CLASS_META[cls]['emoji']} {cls}",
                marker=dict(color=MAP_CLR[cls], size=2.5, opacity=0.55, line=dict(width=0)),
                hovertemplate=f"<b>{cls}</b><br>l: %{{lon:.1f}}°<br>b: %{{lat:.1f}}°<extra></extra>",
            ))

    u_lon, u_b = _to_galactic(user_alpha, user_delta)
    fig.add_trace(go.Scattergeo(
        lon=[u_lon], lat=[u_b],
        mode='markers',
        name='Your object',
        marker=dict(color='white', size=13, symbol='star',
                    line=dict(color='#78aaf8', width=1.5), opacity=1.0),
        hovertemplate=f"RA {user_alpha:.2f}°  Dec {user_delta:.2f}°<extra>Your object</extra>",
    ))

    fig.update_geos(
        projection_type='mollweide',
        showland=False, showocean=False, showlakes=False,
        showrivers=False, showcoastlines=False,
        showframe=True, framecolor='rgba(65,105,185,0.45)',
        bgcolor='#010408',
        lataxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.07)', dtick=30),
        lonaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.07)', dtick=60),
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#4a6890', family='Inter', size=10),
        legend=dict(
            bgcolor='rgba(2,6,20,0.92)',
            bordercolor='rgba(65,105,185,0.3)', borderwidth=1,
            font=dict(size=10, color='#6a8ab0'),
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
        ),
        margin=dict(t=40, b=10, l=10, r=10),
        height=400, hovermode='closest',
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# LANDING / HERO
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state['show_app']:
    # All text + cards as pure HTML so CSS centering works perfectly
    st.markdown("""
    <div class="hero-section">
      <div class="hero-eyebrow">Sloan Digital Sky Survey · SDSS</div>

      <div class="hero-title">Stellar Object<br><span>Classifier</span></div>

      <p class="hero-desc">
        A machine-learning model trained on <b style="color:#a0c8ff">577 000</b> labelled
        photometric observations to distinguish three types of cosmic objects -
        galaxies, stars and quasi-stellar objects - using spectroscopic and photometric features.
      </p>

      <div class="hero-stats">
        <div class="stat-item"><b>577K</b>observations</div>
        <div class="stat-item"><b>95.35%</b>balanced accuracy</div>
        <div class="stat-item"><b>3</b>object classes</div>
        <div class="stat-item"><b>11</b>features</div>
      </div>

      <div class="class-grid">
        <div class="class-card">
          <div class="class-emoji">🌌</div>
          <div class="class-name">GALAXY</div>
          <div class="class-desc">Vast systems of billions of stars, gas and dark matter bound by gravity</div>
        </div>
        <div class="class-card">
          <div class="class-emoji">⭐</div>
          <div class="class-name">STAR</div>
          <div class="class-desc">Luminous plasma balls fusing hydrogen into helium in their cores</div>
        </div>
        <div class="class-card">
          <div class="class-emoji">✨</div>
          <div class="class-name">QSO</div>
          <div class="class-desc">Hyper-luminous active galactic nuclei powered by supermassive black holes</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Inject CTA button style - only active on landing page
    st.markdown(textwrap.dedent("""\
<style>
@keyframes ctaShimmer {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
.stButton > button {
  background: linear-gradient(270deg,#1a3f9a,#2a60d8,#4a90ff,#2a60d8,#1a3f9a) !important;
  background-size: 400% 400% !important;
  animation: ctaShimmer 3.5s ease infinite !important;
  border: 1px solid rgba(140,195,255,0.6) !important;
  border-radius: 100px !important;
  font-size: 1.2rem !important;
  font-weight: 700 !important;
  padding: 1.1rem 2.5rem !important;
  letter-spacing: 0.08em !important;
  box-shadow: 0 8px 32px rgba(50,130,255,0.5) !important;
  width: 100% !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}
.stButton > button * {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  background: transparent !important;
}
.stButton > button:hover {
  background: linear-gradient(270deg,#1f4db8,#3575ee,#60aaff,#3575ee,#1f4db8) !important;
  background-size: 400% 400% !important;
  animation: ctaShimmer 2s ease infinite !important;
  box-shadow: 0 12px 44px rgba(60,150,255,0.65) !important;
  transform: translateY(-2px) !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}
.stButton > button:hover * {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  background: transparent !important;
}
</style>
"""), unsafe_allow_html=True)

    _, btn_col, _ = st.columns([1.8, 1, 1.8])
    with btn_col:
        if st.button("Begin Analysis  →", use_container_width=True):
            st.session_state['show_app'] = True
            st.rerun()

    # ── Methodology section ───────────────────────────────────────────────────
    st.markdown(textwrap.dedent("""\
<div class="meth-wrap">
<div class="meth-title">
<div class="meth-eyebrow">under the hood</div>
How It Works
</div>
<div class="meth-card" style="margin-bottom:1.2rem;">
<div class="mc-hdr">ML Pipeline</div>
<div class="pipe-flow">
<div class="pipe-step">
<div class="pipe-ico">📊</div>
<div class="pipe-body">
<div class="pipe-name">SDSS Input</div>
<div class="pipe-desc">577 K observations · α, δ, u g r i z, redshift, spectral_type, galaxy_population</div>
</div>
</div>
<div class="pipe-arrow">↓</div>
<div class="pipe-step pipe-accent">
<div class="pipe-ico">⚙</div>
<div class="pipe-body">
<div class="pipe-name">FeatureEngineering <span class="pipe-badge">custom transformer</span></div>
<div class="pipe-desc">Derives colour indices from adjacent photometric bands</div>
<div class="pipe-tags"><span class="ptag">u−g</span><span class="ptag">g−r</span><span class="ptag">r−i</span><span class="ptag">i−z</span></div>
</div>
</div>
<div class="pipe-arrow">↓</div>
<div class="pipe-step">
<div class="pipe-ico">🔧</div>
<div class="pipe-body">
<div class="pipe-name">ColumnTransformer</div>
<div class="pipe-split">
<div class="pipe-branch"><span class="branch-label">numeric</span>outlier → NaN · median imputation</div>
<div class="pipe-branch"><span class="branch-label">categorical</span>most-freq impute · OneHotEncoder</div>
</div>
</div>
</div>
<div class="pipe-arrow">↓</div>
<div class="pipe-step pipe-ensemble-step">
<div class="pipe-ico">🤖</div>
<div class="pipe-body">
<div class="pipe-name">Soft Voting Ensemble</div>
<div class="ensemble-chips">
<div class="echip echip-xgb"><div class="echip-name">XGBoost</div><div class="echip-sub">tuned · 50 Optuna trials<br>CV 0.9552</div></div>
<div class="echip-plus">+</div>
<div class="echip echip-lgbm"><div class="echip-name">LightGBM</div><div class="echip-sub">tuned · 50 Optuna trials<br>CV 0.9561</div></div>
<div class="echip-plus">+</div>
<div class="echip echip-cat"><div class="echip-name">CatBoost</div><div class="echip-sub">baseline<br>CV 0.9534</div></div>
</div>
</div>
</div>
<div class="pipe-arrow">↓</div>
<div class="pipe-step pipe-output">
<div class="pipe-ico">✅</div>
<div class="pipe-body">
<div class="pipe-name">Multiclass Prediction</div>
<div class="pipe-tags"><span class="ptag ptag-g">🌌 GALAXY</span><span class="ptag ptag-s">⭐ STAR</span><span class="ptag ptag-q">✨ QSO</span></div>
</div>
</div>
</div>
</div>
<div class="meth-two-col">
<div class="meth-card">
<div class="mc-hdr">Evaluation Strategy</div>
<div class="kf-title">5-fold Stratified K-Fold</div>
<div class="kf-desc">Each fold preserves the class ratio across Galaxy / Star / QSO</div>
<div class="kfold-grid">
<div class="kf-row"><div class="kf-cell kf-test"></div><div class="kf-cell kf-train"></div><div class="kf-cell kf-train"></div><div class="kf-cell kf-train"></div><div class="kf-cell kf-train"></div></div>
<div class="kf-row"><div class="kf-cell kf-train"></div><div class="kf-cell kf-test"></div><div class="kf-cell kf-train"></div><div class="kf-cell kf-train"></div><div class="kf-cell kf-train"></div></div>
<div class="kf-row"><div class="kf-cell kf-train"></div><div class="kf-cell kf-train"></div><div class="kf-cell kf-test"></div><div class="kf-cell kf-train"></div><div class="kf-cell kf-train"></div></div>
<div class="kf-row"><div class="kf-cell kf-train"></div><div class="kf-cell kf-train"></div><div class="kf-cell kf-train"></div><div class="kf-cell kf-test"></div><div class="kf-cell kf-train"></div></div>
<div class="kf-row"><div class="kf-cell kf-train"></div><div class="kf-cell kf-train"></div><div class="kf-cell kf-train"></div><div class="kf-cell kf-train"></div><div class="kf-cell kf-test"></div></div>
</div>
<div class="kf-legend"><span class="kf-leg-test">■ Validation</span><span class="kf-leg-train">■ Train</span></div>
<div class="kf-metric">Metric: <b>Balanced Accuracy</b> - weighted equally across all 3 classes regardless of distribution</div>
</div>
<div class="meth-card">
<div class="mc-hdr">Model Scores (CV Balanced Accuracy)</div>
<div class="scores-list">
<div class="score-row"><div class="score-name">XGBoost baseline</div><div class="score-bar-wrap"><div class="score-bar" style="width:60%"></div></div><div class="score-val">0.9534</div></div>
<div class="score-row"><div class="score-name">XGBoost tuned</div><div class="score-bar-wrap"><div class="score-bar" style="width:75%"></div></div><div class="score-val">0.9552</div></div>
<div class="score-row"><div class="score-name">LightGBM tuned</div><div class="score-bar-wrap"><div class="score-bar" style="width:90%"></div></div><div class="score-val">0.9561</div></div>
<div class="score-row"><div class="score-name">CatBoost baseline</div><div class="score-bar-wrap"><div class="score-bar" style="width:60%"></div></div><div class="score-val">0.9534</div></div>
<div class="score-row score-winner"><div class="score-name">🏆 Ensemble</div><div class="score-bar-wrap"><div class="score-bar score-bar-top" style="width:68%"></div></div><div class="score-val">0.9535</div></div>
</div>
</div>
</div>
</div>
"""), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# APP / CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
else:
    hdr_col, back_col = st.columns([6, 1])
    with hdr_col:
        st.markdown("""
        <div class="app-header">
          <div class="app-header-title">Stellar Classifier</div>
          <div class="app-header-sub">Enter object parameters below or load a known example to begin.</div>
        </div>
        """, unsafe_allow_html=True)
    with back_col:
        st.write("")
        if st.button("← back", use_container_width=True):
            st.session_state['show_app'] = False
            st.session_state['result']   = None
            st.rerun()

    # ── Sky Map ──────────────────────────────────────────────────────────────
    _res     = st.session_state.get('result')
    _u_alpha = _res['alpha'] if _res else float(st.session_state['alpha'])
    _u_delta = _res['delta'] if _res else float(st.session_state['delta'])

    st.markdown('<div class="panel-title" style="margin-bottom:0.4rem;">sky map - sdss footprint</div>', unsafe_allow_html=True)
    st.plotly_chart(make_sky_map(_u_alpha, _u_delta), use_container_width=True)
    st.markdown('<div class="map-footer">4 500-point SDSS sample · ★ marks your input object · hover for coordinates</div>', unsafe_allow_html=True)

    # ── Presets ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-sep">quick load example</div>', unsafe_allow_html=True)
    pb1, pb2, pb3 = st.columns(3)
    _preset = None
    with pb1:
        if st.button("🌌  Galaxy", use_container_width=True): _preset = 'GALAXY'
    with pb2:
        if st.button("⭐  Star",   use_container_width=True): _preset = 'STAR'
    with pb3:
        if st.button("✨  Quasar", use_container_width=True): _preset = 'QSO'
    if _preset:
        for k, v in PRESETS[_preset].items():
            st.session_state[k] = v
        st.session_state['result'] = None
        st.rerun()

    st.markdown('<div class="div-line" style="margin-top:0.8rem;"></div>', unsafe_allow_html=True)

    # ── Inputs + Results ─────────────────────────────────────────────────────
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        # Coordinates
        st.markdown('<div class="sec-lbl-inline">sky coordinates</div>', unsafe_allow_html=True)
        ca, cd = st.columns(2)
        with ca:
            st.markdown('<div class="inp-lbl">α  right ascension (0–360°)</div>', unsafe_allow_html=True)
            alpha = st.number_input("alpha", min_value=0.0, max_value=360.0,
                                    format="%.4f", label_visibility="collapsed", key='alpha')
        with cd:
            st.markdown('<div class="inp-lbl">δ  declination (−20–70°)</div>', unsafe_allow_html=True)
            delta = st.number_input("delta", min_value=-20.0, max_value=70.0,
                                    format="%.4f", label_visibility="collapsed", key='delta')
        st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)

        # Bands
        st.markdown('<div class="panel"><div class="panel-title">photometric magnitudes</div>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        with b1:
            st.markdown('<div class="inp-lbl">u · ultraviolet</div>', unsafe_allow_html=True)
            u = st.number_input("u", format="%.4f", label_visibility="collapsed", key='u')
            st.markdown('<div class="inp-lbl">r · red</div>', unsafe_allow_html=True)
            r = st.number_input("r", format="%.4f", label_visibility="collapsed", key='r')
        with b2:
            st.markdown('<div class="inp-lbl">g · green</div>', unsafe_allow_html=True)
            g = st.number_input("g", format="%.4f", label_visibility="collapsed", key='g')
            st.markdown('<div class="inp-lbl">i · near-IR</div>', unsafe_allow_html=True)
            i_band = st.number_input("i", format="%.4f", label_visibility="collapsed", key='i_band')
        with b3:
            st.markdown('<div class="inp-lbl">z · near-IR</div>', unsafe_allow_html=True)
            z = st.number_input("z", format="%.4f", label_visibility="collapsed", key='z')
        st.markdown('</div>', unsafe_allow_html=True)

        # Redshift
        st.markdown('<div class="inp-lbl">spectroscopic redshift (0 – 7)</div>', unsafe_allow_html=True)
        redshift = st.number_input("redshift", min_value=0.0, max_value=7.0,
                                   step=0.001, format="%.3f",
                                   label_visibility="collapsed", key='redshift')

        # Categorical
        st.markdown('<div class="panel"><div class="panel-title">object metadata</div>', unsafe_allow_html=True)
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown('<div class="inp-lbl">spectral type</div>', unsafe_allow_html=True)
            spectral_type = st.selectbox("Spectral Type", SPEC_OPTS,
                                         label_visibility="collapsed", key='spectral_type')
        with sc2:
            st.markdown('<div class="inp-lbl">galaxy population</div>', unsafe_allow_html=True)
            galaxy_population = st.selectbox("Galaxy Population", POP_OPTS,
                                              label_visibility="collapsed", key='galaxy_population')
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        classify_btn = st.button("classify object →", use_container_width=True)

        if classify_btn:
            input_df = pd.DataFrame([{
                'alpha': alpha, 'delta': delta,
                'u': u, 'g': g, 'r': r, 'i': i_band, 'z': z,
                'redshift': redshift,
                'spectral_type': spectral_type,
                'galaxy_population': galaxy_population,
            }])
            pred  = pipeline.predict(input_df)[0]
            proba = pipeline.predict_proba(input_df)[0]
            label = le.inverse_transform([pred])[0]
            st.session_state['result'] = {
                'label': label, 'proba': proba.tolist(),
                'classes': le.classes_.tolist(),
                'u': u, 'g': g, 'r': r, 'i': i_band, 'z': z,
                'alpha': alpha, 'delta': delta,
            }
            st.rerun()

    with col_out:
        res = st.session_state.get('result')
        if res:
            label   = res['label']
            proba   = np.array(res['proba'])
            classes = res['classes']
            meta    = CLASS_META[label]
            conf    = float(np.max(proba)) * 100

            st.markdown(f"""
            <div class="result-wrap">
              <div class="result-box">
                <span class="result-icon">{meta['emoji']}</span>
                <div class="result-lbl">{label}</div>
                <div class="result-desc">{meta['desc']}</div>
                <div class="result-conf">confidence · <b>{conf:.1f}%</b></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
            st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:0.6rem;letter-spacing:0.22em;color:#4a6888;text-transform:uppercase;margin-bottom:0.5rem;">class probabilities</div>', unsafe_allow_html=True)
            st.markdown(prob_bars_html(classes, proba), unsafe_allow_html=True)

            ci = {'u−g': res['u']-res['g'], 'g−r': res['g']-res['r'],
                  'r−i': res['r']-res['i'], 'i−z': res['i']-res['z']}
            st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
            st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:0.6rem;letter-spacing:0.22em;color:#4a6888;text-transform:uppercase;margin-bottom:0.5rem;">color indices</div>', unsafe_allow_html=True)
            chips = ''.join(f'<div class="ci-chip"><span class="ci-v">{v:+.3f}</span>{k}</div>' for k, v in ci.items())
            st.markdown(f'<div class="ci-grid">{chips}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="await-box">
              <span class="await-ic">◎</span>
              <div class="await-h">awaiting input</div>
              <div class="await-s">enter parameters or load a preset, then press classify</div>
            </div>
            """, unsafe_allow_html=True)
