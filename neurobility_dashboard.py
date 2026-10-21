# Orion.app
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import io

# ── Page config ─────────────────────────────────────────────
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as _SkRF
from sklearn.linear_model import LogisticRegression as _SkLR
class _TrainOnlyImpute:
    def impute(self, X):
        return pd.DataFrame(X).pipe(lambda d: d.mask(d.isna(), self.fill_, axis=1))
    def predict(self, X):
        return super().predict(self.impute(X))
    def predict_proba(self, X):
        return super().predict_proba(self.impute(X))
class RandomForestClassifier(_TrainOnlyImpute, _SkRF): pass
class LogisticRegression(_TrainOnlyImpute, _SkLR):
    def decision_function(self, X):
        return super().decision_function(self.impute(X))

st.set_page_config(
    page_title="ORION v4 — AD Risk Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0f1e;
    color: #e2e8f0;
}

/* Header */
.orion-header {
    background: linear-gradient(135deg, #0a0f1e 0%, #0f1f3d 50%, #0a1628 100%);
    border-bottom: 1px solid rgba(99, 179, 237, 0.2);
    padding: 24px 48px 20px;
    margin: -1rem -1rem 0
}
.orion-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 32px;
    color: #63b3ed;
    letter-spacing: -0.5px;
    margin: 0;
}
.orion-header .subtitle {
    font-size: 13px;
    color: #718096;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0f1f3d;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    border: 1px solid rgba(99,179,237,0.15);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #718096;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 500;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,179,237,0.15) !important;
    color: #63b3ed !important;
}

/* KPI cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin: 16px 0 24px;
}
.kpi-card {
    background: #0f1f3d;
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.kpi-card .kpi-number {
    font-family: 'DM Serif Display', serif;
    font-size: 38px;
    line-height: 1;
    margin-bottom: 4px;
}
.kpi-card .kpi-label {
    font-size: 11px;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500;
}
.kpi-blue   .kpi-number { color: #63b3ed; }
.kpi-red    .kpi-number { color: #fc8181; }
.kpi-amber  .kpi-number { color: #f6ad55; }
.kpi-green  .kpi-number { color: #68d391; }
.kpi-purple .kpi-number { color: #b794f4; }

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.5px;
}
.badge-high     { background: rgba(252,129,129,0.15); color: #fc8181; border: 1px solid rgba(252,129,129,0.3); }
.badge-moderate { background: rgba(246,173,85,0.15);  color: #f6ad55; border: 1px solid rgba(246,173,85,0.3); }
.badge-low      { background: rgba(104,211,145,0.15); color: #68d391; border: 1px solid rgba(104,211,145,0.3); }

/* Result display */
.result-panel {
    border-radius: 14px;
    padding: 24px 28px;
    margin: 16px 0;
    border-left: 4px solid;
}
.result-high     { background: rgba(252,129,129,0.08); border-color: #fc8181; }
.result-moderate { background: rgba(246,173,85,0.08);  border-color: #f6ad55; }
.result-low      { background: rgba(104,211,145,0.08); border-color: #68d391; }
.result-prob {
    font-family: 'DM Serif Display', serif;
    font-size: 56px;
    line-height: 1;
    margin-bottom: 4px;
}
.result-tier {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}

/* Table styling */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Data editor */
.stDataEditor { border-radius: 10px; }

/* Section headers */
.section-head {
    font-family: 'DM Serif Display', serif;
    font-size: 20px;
    color: #63b3ed;
    margin: 24px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(99,179,237,0.15);
}

/* Feature importance bars */
.fi-bar-container { margin: 4px 0; }
.fi-bar-label { font-size: 12px; color: #a0aec0; margin-bottom: 2px; font-family: 'JetBrains Mono', monospace; }
.fi-bar-track { background: rgba(99,179,237,0.1); border-radius: 4px; height: 8px; }
.fi-bar-fill  { background: linear-gradient(90deg, #63b3ed, #b794f4); border-radius: 4px; height: 8px; }

/* Info box */
.info-box {
    background: rgba(99,179,237,0.07);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13px;
    color: #a0aec0;
    margin: 12px 0;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1e3a5f, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(37,99,235,0.4) !important;
}

/* Download button */
.stDownloadButton > button {
    background: rgba(104,211,145,0.1) !important;
    color: #68d391 !important;
    border: 1px solid rgba(104,211,145,0.3) !important;
    border-radius: 8px !important;
}

/* Disclaimer */
.disclaimer {
    background: rgba(246,173,85,0.06);
    border: 1px solid rgba(246,173,85,0.2);
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 11px;
    color: #d69e2e;
    margin-top: 8px;
    font-family: 'JetBrains Mono', monospace;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0f1e;
    border-right: 1px solid rgba(99,179,237,0.1);
}

/* Inputs */
.stNumberInput input, .stTextInput input, .stSelectbox select {
    background: #0f1f3d !important;
    border-color: rgba(99,179,237,0.2) !important;
    color: #e2e8f0 !important;
}

/* Metric */
[data-testid="stMetric"] {
    background: #0f1f3d;
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 10px;
    padding: 12px 16px;
}

/* Progress bar track */
.stProgress > div > div { background: #63b3ed !important; }

</style>
""", unsafe_allow_html=True)

# ── Load model artifacts ─────────────────────────────────────
PACKAGE = Path.home() / "Downloads" / "ORION_v4_package"

@st.cache_resource(show_spinner="Loading ORION model...")
def load_artifacts():
    with open(PACKAGE / "rf_tuned.pkl", "rb") as f:
        model = pickle.load(f)
    with open(PACKAGE / "feature_columns.pkl", "rb") as f:
        features = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    return model, features, explainer

try:
    model, features, explainer = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# ── Feature importance from model ───────────────────────────
@st.cache_data
def get_feature_importance():
    if not model_loaded:
        return {}
    importances = model.feature_importances_
    feat_imp = dict(zip(features, importances))
    return dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

# ── Helper: build input df from row ─────────────────────────
def build_input_row(row):
    age            = float(row.get("Age", 70))
    gender_val     = 0 if str(row.get("Sex", "Male")).strip().lower() in ["male", "0"] else 1
    education      = float(row.get("Education (years)", 14))
    bmi            = float(row.get("BMI", 26))
    apoe4_val      = 1 if str(row.get("APOE4 Carrier", "No")).strip().lower() in ["yes", "carrier", "1"] else 0
    mmse           = float(row.get("MMSE Score", 26))
    moca           = float(row.get("MoCA Score", 24))
    moca_real_val  = 1 if str(row.get("MoCA Real?", "Yes")).strip().lower() in ["yes", "1"] else 0
    iadl           = float(row.get("FAQ/IADL Score", 3))
    cog_slope      = float(row.get("Cognitive Slope 12m", 0))
    cog_slope_24   = float(row.get("Cognitive Slope 24m", 0))
    hypert_val     = 1 if str(row.get("Hypertension", "No")).strip().lower() in ["yes", "1"] else 0

    d = {
        "age"              : age,
        "gender"           : gender_val,
        "education_years"  : education,
        "BMI"              : bmi,
        "APOE4"            : apoe4_val,
        "mmse_score"       : mmse,
        "moca_score"       : moca,
        "moca_real"        : moca_real_val,
        "iadl_score"       : iadl,
        "cognitive_slope"  : cog_slope,
        "cognitive_slope_24": cog_slope_24,
        "APOE4_x_age"      : apoe4_val * age,
        "age_x_gender"     : age * gender_val,
        "mmse_x_age"       : mmse * age,
        "iadl_x_gender"    : iadl * gender_val,
        "education_x_APOE4": education * apoe4_val,
        "hypertension_1"   : hypert_val,
    }
    df = pd.DataFrame([d])
    for col in features:
        if col not in df.columns:
            df[col] = 0
    return df[features]

def tier_badge(tier):
    t = str(tier).strip()
    cls = {"High": "badge-high", "Moderate": "badge-moderate", "Low": "badge-low"}.get(t, "badge-low")
    return f'<span class="risk-badge {cls}">{t.upper()}</span>'

def progression_label(prob):
    return "Fast" if prob >= 0.66 else ("Moderate" if prob >= 0.33 else "Slow")

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="orion-header">
  <h1>🧠 ORION v4</h1>
  <div class="subtitle">Alzheimer's Disease Risk Intelligence Platform &nbsp;|&nbsp; Random Forest · AUC 0.938 · 819 patients · ADNI</div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Could not load model from `{PACKAGE}`. Run the ORION training script first.\n\n`{model_error}`")
    st.stop()

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────
if "cohort_df" not in st.session_state:
    st.session_state.cohort_df = pd.DataFrame(columns=[
        "Subject ID", "Age", "Sex", "Education (years)", "BMI",
        "APOE4 Carrier", "MMSE Score", "MoCA Score", "MoCA Real?",
        "FAQ/IADL Score", "Cognitive Slope 12m", "Cognitive Slope 24m", "Hypertension"
    ])
if "results_df" not in st.session_state:
    st.session_state.results_df = None
# ── NURABILITY SPLASH ANIMATION ─────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=JetBrains+Mono:wght@400;600&display=swap');

#nurability-splash {
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    background: #060f24;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    z-index: 9999;
    animation: fadeOutSplash 0.6s ease-in-out 2.6s forwards;
}
.splash-brand {
    font-family: 'DM Serif Display', serif;
    font-size: 52px; color: #63b3ed;
    letter-spacing: 4px;
    animation: riseIn 0.8s ease-out 0.3s both;
}
.splash-powered {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px; color: #4a6fa5;
    letter-spacing: 3px; margin-top: 10px;
    animation: riseIn 0.8s ease-out 0.9s both;
}
.splash-bar {
    width: 0px; height: 2px;
    background: linear-gradient(90deg, #1a2f52, #63b3ed, #1a2f52);
    margin-top: 28px; border-radius: 2px;
    animation: growBar 1.2s ease-out 1.2s forwards;
}
@keyframes riseIn {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes growBar {
    from { width: 0px; }
    to   { width: 220px; }
}
@keyframes fadeOutSplash {
    from { opacity: 1; pointer-events: all; }
    to   { opacity: 0; pointer-events: none; }
}

/* PERSISTENT HEADER */
.nurability-header {
    font-family: 'DM Serif Display', serif;
    font-size: 13px; color: #4a6fa5;
    letter-spacing: 2px;
    text-align: right;
    padding: 6px 0 18px 0;
    border-bottom: 1px solid #1a2f52;
    margin-bottom: 24px;
}
.nurability-header span {
    color: #63b3ed;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
}
</style>

<div id="nurability-splash">
  <div class="splash-brand">Nurability</div>
  <div class="splash-powered">powered by ORION V4</div>
  <div class="splash-bar"></div>
</div>
""", unsafe_allow_html=True)

# Persistent header that stays after splash fades
st.markdown("""
<div class="nurability-header">
  Nurability &nbsp;·&nbsp; <span>powered by ORION V4</span>
</div>
""", unsafe_allow_html=True)
# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋  Enter Cohort",
    "📊  Cohort Results",
    "🔬  Individual Analysis",
    "⚙️  Model Intelligence",
    "ℹ️  About ORION"
])

# ═══════════════════════════════════════════════════════════
# TAB 1 — ENTER COHORT
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-head">Patient Data Entry</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    Enter patients directly in the table below — add as many rows as needed.
    Or upload a CSV matching the column headers. All fields are editable inline.
    Hit <b>Run ORION Model</b> when ready.
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_template = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"],
                                     label_visibility="collapsed")
        if uploaded:
            try:
                uploaded_df = pd.read_csv(uploaded)
                # Map common column name variants
                rename_map = {
                    "subject_id": "Subject ID", "patient_id": "Subject ID",
                    "age": "Age", "sex": "Sex", "gender": "Sex",
                    "education_years": "Education (years)", "education": "Education (years)",
                    "bmi": "BMI", "apoe4": "APOE4 Carrier",
                    "mmse_score": "MMSE Score", "mmse": "MMSE Score",
                    "moca_score": "MoCA Score", "moca": "MoCA Score",
                    "moca_real": "MoCA Real?",
                    "iadl_score": "FAQ/IADL Score", "faq": "FAQ/IADL Score",
                    "cognitive_slope": "Cognitive Slope 12m",
                    "cognitive_slope_24": "Cognitive Slope 24m",
                    "hypertension": "Hypertension",
                }
                uploaded_df.columns = [rename_map.get(c.lower().strip(), c) for c in uploaded_df.columns]
                st.session_state.cohort_df = uploaded_df
                st.success(f"✓ Loaded {len(uploaded_df)} patients from CSV")
            except Exception as e:
                st.error(f"Upload error: {e}")

    with col_template:
        template = pd.DataFrame([{
            "Subject ID": "PT_001", "Age": 68, "Sex": "Female",
            "Education (years)": 16, "BMI": 24.5, "APOE4 Carrier": "Yes",
            "MMSE Score": 24, "MoCA Score": 22, "MoCA Real?": "Yes",
            "FAQ/IADL Score": 5, "Cognitive Slope 12m": -2.0,
            "Cognitive Slope 24m": -3.5, "Hypertension": "No"
        }])
        st.download_button("⬇ Download Template CSV", template.to_csv(index=False),
                           "orion_template.csv", "text/csv", use_container_width=True)

    # Editable data grid
    default_row = {
        "Subject ID": "", "Age": 68, "Sex": "Female",
        "Education (years)": 14, "BMI": 26.0, "APOE4 Carrier": "No",
        "MMSE Score": 26, "MoCA Score": 24, "MoCA Real?": "Yes",
        "FAQ/IADL Score": 3, "Cognitive Slope 12m": 0.0,
        "Cognitive Slope 24m": 0.0, "Hypertension": "No"
    }

    if st.session_state.cohort_df.empty:
        # Seed with 3 example rows
        st.session_state.cohort_df = pd.DataFrame([
            {"Subject ID": "PT_001", "Age": 72, "Sex": "Female", "Education (years)": 12,
             "BMI": 28.1, "APOE4 Carrier": "Yes", "MMSE Score": 22, "MoCA Score": 20,
             "MoCA Real?": "Yes", "FAQ/IADL Score": 8, "Cognitive Slope 12m": -3.0,
             "Cognitive Slope 24m": -5.0, "Hypertension": "Yes"},
            {"Subject ID": "PT_002", "Age": 65, "Sex": "Male", "Education (years)": 18,
             "BMI": 23.5, "APOE4 Carrier": "No", "MMSE Score": 28, "MoCA Score": 27,
             "MoCA Real?": "Yes", "FAQ/IADL Score": 1, "Cognitive Slope 12m": 0.0,
             "Cognitive Slope 24m": -0.5, "Hypertension": "No"},
            {"Subject ID": "PT_003", "Age": 78, "Sex": "Female", "Education (years)": 10,
             "BMI": 31.2, "APOE4 Carrier": "Yes", "MMSE Score": 19, "MoCA Score": 17,
             "MoCA Real?": "Yes", "FAQ/IADL Score": 14, "Cognitive Slope 12m": -4.5,
             "Cognitive Slope 24m": -7.0, "Hypertension": "Yes"},
        ])

    edited_df = st.data_editor(
        st.session_state.cohort_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Subject ID"         : st.column_config.TextColumn("Subject ID", width="small"),
            "Age"                : st.column_config.NumberColumn("Age", min_value=40, max_value=100, step=1),
            "Sex"                : st.column_config.SelectboxColumn("Sex", options=["Male", "Female"]),
            "Education (years)"  : st.column_config.NumberColumn("Education (yrs)", min_value=0, max_value=25, step=1),
            "BMI"                : st.column_config.NumberColumn("BMI", min_value=15.0, max_value=50.0, format="%.1f"),
            "APOE4 Carrier"      : st.column_config.SelectboxColumn("APOE4", options=["Yes", "No"]),
            "MMSE Score"         : st.column_config.NumberColumn("MMSE", min_value=0, max_value=30, step=1),
            "MoCA Score"         : st.column_config.NumberColumn("MoCA", min_value=0, max_value=30, step=1),
            "MoCA Real?"         : st.column_config.SelectboxColumn("MoCA Real?", options=["Yes", "No"]),
            "FAQ/IADL Score"     : st.column_config.NumberColumn("FAQ/IADL", min_value=0, max_value=30, step=1),
            "Cognitive Slope 12m": st.column_config.NumberColumn("Cog Slope 12m", min_value=-15.0, max_value=5.0, format="%.1f"),
            "Cognitive Slope 24m": st.column_config.NumberColumn("Cog Slope 24m", min_value=-20.0, max_value=5.0, format="%.1f"),
            "Hypertension"       : st.column_config.SelectboxColumn("Hypertension", options=["Yes", "No"]),
        },
        height=320,
        key="cohort_editor"
    )
    

    st.markdown(f"<div style='font-size:12px;color:#718096;margin-top:4px;'>{len(edited_df)} patient(s) entered</div>", unsafe_allow_html=True)

    col_run, col_clear = st.columns([2, 1])
    with col_run:
        run_btn = st.button("🔬  Run ORION Model on Full Cohort", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear All", use_container_width=True):
            st.session_state.cohort_df = pd.DataFrame(columns=edited_df.columns)
            st.session_state.results_df = None
            st.rerun()

    if run_btn:
        if len(edited_df) == 0:
            st.warning("No patients entered yet.")
        else:
            with st.spinner("Running ORION model..."):
                results = []
                for _, row in edited_df.iterrows():
                    try:
                        input_df = build_input_row(row)
                        prob = model.predict_proba(input_df)[0][1]
                        tier = "High" if prob >= 0.66 else ("Moderate" if prob >= 0.33 else "Low")
                        prog = progression_label(prob)
                        results.append({
                            "Subject ID"        : row.get("Subject ID", "—"),
                            "Age"               : row.get("Age", "—"),
                            "Sex"               : row.get("Sex", "—"),
                            "APOE4"             : row.get("APOE4 Carrier", "—"),
                            "MMSE"              : row.get("MMSE Score", "—"),
                            "AD Risk Prob"      : round(float(prob), 4),
                            "Risk Tier"         : tier,
                            "Progression"       : prog,
                            "_input_row"        : row.to_dict(),
                        })
                    except Exception as e:
                        results.append({
                            "Subject ID": row.get("Subject ID", "ERR"),
                            "AD Risk Prob": None,
                            "Risk Tier": "Error",
                            "Progression": str(e),
                            "_input_row": {},
                        })
                st.session_state.results_df = pd.DataFrame(results)
            st.success(f"✓ ORION scored {len(results)} patients. See **Cohort Results** tab.")

    st.markdown('<div class="disclaimer">⚠️ ORION v4 is a research tool. Outputs are probabilistic risk estimates, not diagnoses. Not for individual clinical decisions.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 2 — COHORT RESULTS
# ═══════════════════════════════════════════════════════════
with tab2:
    if st.session_state.results_df is None or len(st.session_state.results_df) == 0:
        st.markdown("""
        <div style="text-align:center; padding: 80px 0; color: #4a5568;">
          <div style="font-size:48px; margin-bottom:16px;">📊</div>
          <div style="font-family:'DM Serif Display',serif; font-size:22px; color:#63b3ed; margin-bottom:8px;">No results yet</div>
          <div style="font-size:14px;">Enter patients in the <b>Enter Cohort</b> tab and click <b>Run ORION Model</b>.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        rdf = st.session_state.results_df
        valid = rdf[rdf["AD Risk Prob"].notna()]

        n_total    = len(valid)
        n_high     = (valid["Risk Tier"] == "High").sum()
        n_moderate = (valid["Risk Tier"] == "Moderate").sum()
        n_low      = (valid["Risk Tier"] == "Low").sum()
        n_fast     = (valid["Progression"] == "Fast").sum()
        avg_risk   = valid["AD Risk Prob"].mean()

        # KPI row
        st.markdown(f"""
        <div class="kpi-grid">
          <div class="kpi-card kpi-blue">
            <div class="kpi-number">{n_total}</div>
            <div class="kpi-label">Total Patients</div>
          </div>
          <div class="kpi-card kpi-red">
            <div class="kpi-number">{n_high}</div>
            <div class="kpi-label">High Risk</div>
          </div>
          <div class="kpi-card kpi-amber">
            <div class="kpi-number">{n_moderate}</div>
            <div class="kpi-label">Moderate Risk</div>
          </div>
          <div class="kpi-card kpi-green">
            <div class="kpi-number">{n_low}</div>
            <div class="kpi-label">Low Risk</div>
          </div>
          <div class="kpi-card kpi-purple">
            <div class="kpi-number">{avg_risk:.1%}</div>
            <div class="kpi-label">Mean AD Risk</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Charts row
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig1, ax1 = plt.subplots(figsize=(5, 3.2))
            fig1.patch.set_facecolor('#0f1f3d')
            ax1.set_facecolor('#0f1f3d')
            colors_tier = ['#fc8181', '#f6ad55', '#68d391']
            tier_vals   = [n_high, n_moderate, n_low]
            tier_labels = ['High', 'Moderate', 'Low']
            bars = ax1.bar(tier_labels, tier_vals, color=colors_tier, edgecolor='none', width=0.5)
            for bar, val in zip(bars, tier_vals):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                         str(val), ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')
            ax1.set_title('Risk Tier Distribution', color='#63b3ed', fontsize=12, pad=10)
            ax1.tick_params(colors='#718096')
            ax1.spines[:].set_color('#1a2f52')
            ax1.set_ylim(0, max(tier_vals) * 1.25 + 1)
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=True)
            plt.close()

        with col_chart2:
            fig2, ax2 = plt.subplots(figsize=(5, 3.2))
            fig2.patch.set_facecolor('#0f1f3d')
            ax2.set_facecolor('#0f1f3d')
            ax2.hist(valid["AD Risk Prob"], bins=min(20, n_total),
                     color='#63b3ed', edgecolor='#0a0f1e', alpha=0.8)
            ax2.axvline(0.33, color='#f6ad55', linestyle='--', linewidth=1.2, label='Low/Mod')
            ax2.axvline(0.66, color='#fc8181', linestyle='--', linewidth=1.2, label='Mod/High')
            ax2.set_title('AD Risk Probability Distribution', color='#63b3ed', fontsize=12, pad=10)
            ax2.set_xlabel('Risk Probability', color='#718096', fontsize=10)
            ax2.tick_params(colors='#718096')
            ax2.spines[:].set_color('#1a2f52')
            ax2.legend(fontsize=9, framealpha=0.3, labelcolor='#a0aec0')
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        st.markdown('<div class="section-head">Patient Results Table</div>', unsafe_allow_html=True)

        # Render table with colored badges
        display_rows = []
        for _, row in valid.sort_values("AD Risk Prob", ascending=False).iterrows():
            tier = row["Risk Tier"]
            display_rows.append({
                "Subject ID" : row["Subject ID"],
                "Age"        : row["Age"],
                "Sex"        : row["Sex"],
                "APOE4"      : row["APOE4"],
                "MMSE"       : row["MMSE"],
                "AD Risk %"  : f"{row['AD Risk Prob']:.1%}",
                "Risk Tier"  : tier,
                "Progression": row["Progression"],
            })
        display_df = pd.DataFrame(display_rows)

        def color_tier(val):
            colors = {"High": "color: #fc8181; font-weight: 600",
                      "Moderate": "color: #f6ad55; font-weight: 600",
                      "Low": "color: #68d391; font-weight: 600"}
            return colors.get(val, "")

        def color_prog(val):
            return "color: #fc8181; font-weight:600" if val == "Fast" else "color: #68d391"

        styled = display_df.style\
            .map(color_tier, subset=["Risk Tier"])\
.           map(color_prog, subset=["Progression"])\
            .set_properties(**{"background-color": "#0f1f3d", "color": "#e2e8f0",
                               "border-color": "#1a2f52", "font-size": "13px"})

        st.dataframe(styled, use_container_width=True, height=350)

        # Download
        csv_out = valid[["Subject ID", "Age", "Sex", "APOE4",
                          "MMSE", "AD Risk Prob", "Risk Tier", "Progression"]].to_csv(index=False)
        st.download_button("⬇ Download Results CSV", csv_out,
                           "orion_cohort_results.csv", "text/csv")

        st.markdown('<div class="disclaimer">⚠️ Research use only. Results are not diagnostic.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 3 — INDIVIDUAL ANALYSIS
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-head">Individual Patient Deep Dive</div>', unsafe_allow_html=True)

    if st.session_state.results_df is None or len(st.session_state.results_df) == 0:
        st.markdown("""
        <div style="text-align:center; padding: 60px 0; color: #4a5568;">
          <div style="font-size:40px; margin-bottom:12px;">🔬</div>
          <div style="font-size:14px;">Run the model first in the <b>Enter Cohort</b> tab.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        rdf = st.session_state.results_df
        valid = rdf[rdf["AD Risk Prob"].notna()]
        subject_ids = valid["Subject ID"].tolist()

        selected_id = st.selectbox("Select Patient", subject_ids,
                                   format_func=lambda x: f"Patient: {x}")

        sel_row = valid[valid["Subject ID"] == selected_id].iloc[0]
        prob    = float(sel_row["AD Risk Prob"])
        tier    = sel_row["Risk Tier"]
        prog    = sel_row["Progression"]

        tier_class = {"High": "result-high", "Moderate": "result-moderate", "Low": "result-low"}.get(tier, "result-low")
        prob_color = {"High": "#fc8181", "Moderate": "#f6ad55", "Low": "#68d391"}.get(tier, "#68d391")
        emoji      = {"High": "🔴", "Moderate": "🟡", "Low": "🟢"}.get(tier, "🟢")

        col_res, col_detail = st.columns([1, 1])
        with col_res:
            st.markdown(f"""
            <div class="result-panel {tier_class}">
              <div style="font-size:12px;color:#718096;font-family:'JetBrains Mono',monospace;margin-bottom:8px;">AD RISK PROBABILITY</div>
              <div class="result-prob" style="color:{prob_color};">{emoji} {prob:.1%}</div>
              <div class="result-tier" style="color:{prob_color};">{tier} RISK &nbsp;·&nbsp; {prog} PROGRESSION</div>
            </div>
            """, unsafe_allow_html=True)

            # Mini feature table for this patient
            input_data = sel_row.get("_input_row", {})
            if input_data:
                st.markdown("**Patient Profile**")
                profile_data = {k: v for k, v in input_data.items()
                                if k not in ["Subject ID"] and not k.startswith("_")}
                profile_df = pd.DataFrame(list(profile_data.items()), columns=["Feature", "Value"])
                st.dataframe(profile_df, use_container_width=True, hide_index=True, height=200)

        with col_detail:
            st.markdown("**Contributing Risk Factors (SHAP)**")
            st.caption("🔴 Red bars push risk UP &nbsp;|&nbsp; 🔵 Blue bars push risk DOWN")
            try:
                input_df   = build_input_row(sel_row["_input_row"])
                shap_vals  = explainer(input_df)
                fig_shap, ax_shap = plt.subplots(figsize=(6, 5))
                fig_shap.patch.set_facecolor('#0f1f3d')
                shap.plots.waterfall(shap_vals[:, :, 1][0], max_display=12, show=False)
                fig_shap.patch.set_facecolor('#0f1f3d')
                plt.gcf().set_facecolor('#0f1f3d')
                st.pyplot(plt.gcf(), use_container_width=True)
                plt.close('all')
            except Exception as e:
                st.warning(f"SHAP plot unavailable: {e}")

        st.markdown('<div class="disclaimer">⚠️ SHAP values show feature contributions for this individual patient only. Research use only.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 4 — MODEL INTELLIGENCE
# ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-head">ORION Model Intelligence</div>', unsafe_allow_html=True)

    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        st.markdown("**Model Performance**")
        perf_data = {
            "Metric": ["CV AUC (5-fold)", "CV Std Dev", "Test AUC", "Total Training Patients",
                        "AD Patients", "Non-AD Patients", "Algorithm"],
            "Value":  ["0.938", "±0.019", "0.960", "819", "368", "451", "Random Forest (tuned)"]
        }
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True, hide_index=True, height=280)

        st.markdown("**Training Dataset**")
        st.markdown("""
        <div class="info-box">
        <b>Source:</b> Alzheimer's Disease Neuroimaging Initiative (ADNI)<br>
        <b>Dataset:</b> all_subject_AD_ORION1_My_Table_02Mar2026.csv<br>
        <b>Supplemental:</b> MoCA scores, FAQ/IADL, DXSUM (diagnosis)<br>
        <b>Exclusions:</b> Parkinson's dementia, other non-AD dementias<br>
        <b>Validation needed:</b> NACC or OASIS-3 (500+ patients)
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("**Feature Importance (replaces Model Coefficients)**")
        feat_imp = get_feature_importance()
        if feat_imp:
            max_imp = max(feat_imp.values())
            fig_fi, ax_fi = plt.subplots(figsize=(5, 6))
            fig_fi.patch.set_facecolor('#0f1f3d')
            ax_fi.set_facecolor('#0f1f3d')

            feats  = list(reversed(list(feat_imp.keys())[:14]))
            imps   = list(reversed(list(feat_imp.values())[:14]))
            colors = ['#b794f4' if i >= len(feats)-3
                      else '#63b3ed' if i >= len(feats)-8
                      else '#4a6fa5' for i in range(len(feats))]

            bars = ax_fi.barh(feats, imps, color=colors, edgecolor='none', height=0.6)
            ax_fi.set_xlabel("Importance Score", color='#718096', fontsize=10)
            ax_fi.set_title("Feature Importance", color='#63b3ed', fontsize=12, pad=10)
            ax_fi.tick_params(colors='#a0aec0', labelsize=9)
            ax_fi.spines[:].set_color('#1a2f52')
            plt.tight_layout()
            st.pyplot(fig_fi, use_container_width=True)
            plt.close()

    st.markdown('<div class="section-head">Feature Reference</div>', unsafe_allow_html=True)
    feat_ref = pd.DataFrame([
        {"Feature": "mmse_score",        "Description": "Mini-Mental State Examination score (0-30)",          "Source": "ADNI clinical"},
        {"Feature": "moca_score",         "Description": "Montreal Cognitive Assessment score (0-30)",          "Source": "ADNI supplemental"},
        {"Feature": "iadl_score",         "Description": "FAQ Instrumental Activities of Daily Living total",   "Source": "ADNI supplemental"},
        {"Feature": "cognitive_slope",    "Description": "MMSE change from baseline to 12 months",             "Source": "ADNI longitudinal"},
        {"Feature": "cognitive_slope_24", "Description": "MMSE change from baseline to 24 months",             "Source": "ADNI longitudinal"},
        {"Feature": "APOE4",              "Description": "APOE ε4 allele carrier status (0=No, 1=Yes)",        "Source": "ADNI genetic"},
        {"Feature": "APOE4_x_age",        "Description": "APOE4 × Age interaction (risk amplified by age)",    "Source": "Engineered"},
        {"Feature": "age",                "Description": "Patient age at baseline (years)",                    "Source": "ADNI demographic"},
        {"Feature": "mmse_x_age",         "Description": "MMSE × Age interaction",                            "Source": "Engineered"},
        {"Feature": "education_years",    "Description": "Years of formal education (cognitive reserve proxy)", "Source": "ADNI demographic"},
        {"Feature": "BMI",                "Description": "Body Mass Index (kg/m²)",                           "Source": "ADNI clinical"},
        {"Feature": "gender",             "Description": "Sex (0=Male, 1=Female)",                             "Source": "ADNI demographic"},
        {"Feature": "hypertension_1",     "Description": "Hypertension diagnosis (0=No, 1=Yes)",               "Source": "ADNI medical history"},
        {"Feature": "moca_real",          "Description": "Flag: 1 = real MoCA score, 0 = imputed median",      "Source": "Engineered"},
    ])
    st.dataframe(feat_ref, use_container_width=True, hide_index=True, height=400)


# ═══════════════════════════════════════════════════════════
# TAB 5 — ABOUT ORION
# ═══════════════════════════════════════════════════════════
with tab5:
    col_a1, col_a2 = st.columns([3, 2])

    with col_a1:
        st.markdown("""
        <div style="font-family:'DM Serif Display',serif; font-size:28px; color:#63b3ed; margin-bottom:4px;">
        O.R.I.O.N
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:12px; color:#718096; margin-bottom:24px;">
        Optimized Risk Intelligence for Onset Neurodegeneration · v4.0
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <b>System Overview</b><br><br>
        ORION is a machine learning-powered risk intelligence platform designed to stratify
        patients by Alzheimer's Disease risk using clinical, cognitive, genetic, and
        demographic features from the ADNI dataset.<br><br>
        The system applies a tuned Random Forest classifier (CV AUC 0.938) trained on 819
        patients to generate individual AD risk probabilities, risk tier classifications
        (Low / Moderate / High), and progression speed estimates (Slow / Moderate / Fast).<br><br>
        SHAP (SHapley Additive exPlanations) values provide patient-level interpretability,
        showing which features are driving each individual's risk score up or down.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <b>Intended Use</b><br><br>
        ORION is intended for research, clinical trial optimization, and population-level
        risk stratification. Outputs are designed to support cohort segmentation, enrollment
        strategy, and longitudinal analysis.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <b>Use of Illustrative Data</b><br><br>
        Example rows in the Enter Cohort tab are provided solely to demonstrate data schema
        and system workflow. They do not represent real patient data, model validation, or
        predictive accuracy. All pilot and production analyses use externally supplied datasets.
        </div>
        """, unsafe_allow_html=True)

    with col_a2:
        st.markdown("""
        <div style="background:#0f1f3d; border:1px solid rgba(99,179,237,0.15); border-radius:12px; padding:20px 24px;">
        <div style="font-size:11px;color:#718096;text-transform:uppercase;letter-spacing:1px;margin-bottom:16px;font-family:'JetBrains Mono',monospace;">Product Information</div>
        """, unsafe_allow_html=True)

        info_items = [
            ("Product", "O.R.I.O.N"),
            ("Version", "v4.0"),
            ("Owner", "Amir Kelson"),
            ("Contact", "akelson@xula.edu"),
            ("Model", "Random Forest (tuned)"),
            ("CV AUC", "0.938 ± 0.019"),
            ("Test AUC", "0.960"),
            ("Training n", "819 patients"),
            ("Dataset", "ADNI"),
            ("Features", "16 + interactions"),
            ("Date", "April 2026"),
        ]
        for label, val in info_items:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(99,179,237,0.08);">
              <span style="font-size:12px;color:#718096;">{label}</span>
              <span style="font-size:12px;color:#e2e8f0;font-family:'JetBrains Mono',monospace;">{val}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(252,129,129,0.06); border:1px solid rgba(252,129,129,0.2); border-radius:10px; padding:16px 20px;">
        <div style="font-size:11px;color:#fc8181;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">⚠️ Research Use Only</div>
        <div style="font-size:12px;color:#a0aec0;line-height:1.6;">
        ORION outputs represent probabilistic risk stratification. They are <b>not diagnostic</b>,
        not clinical advice, and not intended for individual patient treatment decisions.
        Always consult qualified clinicians for medical decisions.
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>Recommended Next Steps</b><br><br>
        • External validation on NACC or OASIS-3 (500+ patients)<br>
        • Sex-stratified validation (male/female models available)<br>
        • Longitudinal follow-up validation<br>
        • IRB review before clinical deployment
        </div>
        """, unsafe_allow_html=True)