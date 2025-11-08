# --------------------------------------------------------------
# Synthetic Traffic & Transportation Data Generator (STTDG+)
# Version: 1.3 (robust, research-grade)
# --------------------------------------------------------------
# Features
# - Datasets: Traffic Flow (static & time series), Incidents, Accidents, Driver Behavior
# - Fundamental diagrams (Greenshields, Greenberg, Underwood)
# - Diurnal demand curves by day type (weekday/weekend) + weather multipliers
# - Non-Homogeneous Poisson Process (NHPP) for incident arrivals
# - Gaussian copula for correlated variables (behavioral + accident covariates)
# - Scenario presets + full parameter controls
# - Validation: distribution checks, correlations, missingness, bounds
# - Downloads: CSV + JSON metadata (provenance & parameter record)
# --------------------------------------------------------------

import json
import io
import time
from dataclasses import asdict, dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

# ------------------------- Page -------------------------
st.set_page_config(
    page_title="STTDG+ | Synthetic Traffic & Transportation Data Generator",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------- Style -------------------------
st.markdown("""
<style>
    .main { background: #f7f9fc; }
    h1, h2, h3 { color: #003a6f; }
    .stButton > button {
        background:#003a6f; color:white; border-radius:10px;
        padding:0.5rem 1rem; font-weight:600; border:0;
    }
    .stButton > button:hover { background:#00509e; }
    .param-card {
        background: white; padding: 1rem 1.2rem; border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05); border: 1px solid #eef1f6;
    }
    .good { color:#2e7d32; font-weight:600; }
    .warn { color:#ef6c00; font-weight:600; }
    .bad { color:#c62828; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ------------------------- Helpers -------------------------
def _clip(a, lo, hi):
    return np.minimum(np.maximum(a, lo), hi)

def _download_bytes(name: str, content: bytes, label: str, mime: str):
    st.download_button(label, data=content, file_name=name, mime=mime)

def _png_from_matplotlib(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def _nhpp_thinning(rate_t: np.ndarray, horizon_h: int, seed: int) -> np.ndarray:
    """
    Non-Homogeneous Poisson Process via thinning.
    rate_t: array of rates (lambda per hour) of length horizon_h
    Returns event counts per hour (int array length horizon_h).
    """
    rng = np.random.default_rng(seed)
    lam_max = float(np.max(rate_t))
    counts = np.zeros(horizon_h, dtype=int)
    # simulate homogeneous PP with lambda_max, then thin
    n_total = rng.poisson(lam_max * horizon_h)
    # propose event times uniformly in [0, horizon_h)
    t_prop = rng.uniform(0, horizon_h, size=n_total)
    # acceptance based on rate(t)/lam_max
    t_sorted = np.sort(t_prop)
    acc = []
    for t in t_sorted:
        idx = min(int(np.floor(t)), horizon_h-1)
        if rng.uniform() < (rate_t[idx] / lam_max if lam_max > 0 else 0):
            acc.append(idx)
    # count by hour
    for i in acc:
        counts[i] += 1
    return counts

def _gaussian_copula(n: int, means: np.ndarray, stds: np.ndarray, corr: np.ndarray, seed: int) -> np.ndarray:
    """
    Gaussian copula sampler: returns correlated normal samples with given means/stds/corr
    """
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal(size=(n, corr.shape[0])) @ L.T   # correlated standard normals
    x = z * stds + means
    return x

# ------------------------- Metadata -------------------------
@dataclass
class Meta:
    app_name: str
    version: str
    created_utc: float
    seed: int
    dataset_type: str
    params: Dict[str, Any]

def _pack_metadata(dataset_type: str, seed: int, params: Dict[str, Any]) -> bytes:
    meta = Meta(
        app_name="STTDG+",
        version="1.3",
        created_utc=time.time(),
        seed=seed,
        dataset_type=dataset_type,
        params=params
    )
    return json.dumps(asdict(meta), indent=2).encode("utf-8")

# ------------------------- Sidebar -------------------------
st.sidebar.header("⚙️ Generator Settings")
dataset_type = st.sidebar.selectbox(
    "Dataset",
    ["Traffic Flow (Static)", "Traffic Flow (Time Series, 24h)", "Incidents (Events)", "Accidents (Cases)", "Driver Behavior (Cross-Section)"]
)

seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, value=2025, step=1)
rng = np.random.default_rng(seed)

st.sidebar.markdown("### Presets")
preset = st.sidebar.selectbox(
    "Scenario",
    ["Default / Balanced", "Urban Peak-Constrained", "Rural Free-Flow", "Rainy Congestion", "Incident-Prone Corridor"]
)

st.sidebar.markdown("---")
show_plots = st.sidebar.checkbox("Show visualizations", True)
show_validation = st.sidebar.checkbox("Show validation summaries", True)

# ---------- Preset logic ----------
preset_params: Dict[str, Any] = {}
if preset == "Default / Balanced":
    preset_params = dict(vf=95, kj=150, base_vph=1600, ts_noise=0.08, weather="Clear")
elif preset == "Urban Peak-Constrained":
    preset_params = dict(vf=75, kj=180, base_vph=2200, ts_noise=0.12, weather="Clear")
elif preset == "Rural Free-Flow":
    preset_params = dict(vf=110, kj=110, base_vph=1200, ts_noise=0.06, weather="Clear")
elif preset == "Rainy Congestion":
    preset_params = dict(vf=80, kj=170, base_vph=2100, ts_noise=0.10, weather="Rainy")
elif preset == "Incident-Prone Corridor":
    preset_params = dict(vf=90, kj=160, base_vph=1800, ts_noise=0.09, weather="Clear")

# ------------------------- Parameter Cards -------------------------
st.markdown("### Parameters")
cols = st.columns(3)

with cols[0]:
    st.markdown("#### Fundamental Diagram", help="Used in flow generation")
    fd_model = st.selectbox("Model", ["Greenshields", "Greenberg", "Underwood"], index=0)
    vf = st.number_input("Free-flow speed vf (km/h)", value=int(preset_params.get("vf", 95)), min_value=40, max_value=140, step=1)
    kj = st.number_input("Jam density kj (veh/km)", value=int(preset_params.get("kj", 150)), min_value=80, max_value=220, step=5)

with cols[1]:
    st.markdown("#### Diurnal Demand", help="Only for Time Series generation")
    base_vph = st.number_input("Base hourly demand (veh/h)", value=int(preset_params.get("base_vph", 1600)), min_value=200, max_value=4000, step=50)
    day_type = st.selectbox("Day type", ["Weekday", "Weekend"], index=0)
    ts_noise = st.slider("Time-series noise (σ, relative)", min_value=0.0, max_value=0.5, value=float(preset_params.get("ts_noise", 0.08)), step=0.01)

with cols[2]:
    st.markdown("#### Exogenous Conditions")
    weather = st.selectbox("Weather", ["Clear", "Rainy", "Foggy"], index=["Clear","Rainy","Foggy"].index(preset_params.get("weather","Clear")))
    weather_flow_multiplier = {"Clear": 1.00, "Rainy": 0.93, "Foggy": 0.90}[weather]
    weather_incident_multiplier = {"Clear": 1.00, "Rainy": 1.35, "Foggy": 1.50}[weather]
    st.write(f"- Flow multiplier: **{weather_flow_multiplier:.2f}**")
    st.write(f"- Incident multiplier: **{weather_incident_multiplier:.2f}**")

# Size controls by dataset
st.markdown("---")
if dataset_type == "Traffic Flow (Static)":
    n = st.slider("Number of sections (records)", 100, 50_000, 3000, step=100)
elif dataset_type == "Traffic Flow (Time Series, 24h)":
    n_sections = st.slider("Number of sections (links)", 10, 1000, 100, step=10)
    horizon_h = 24
elif dataset_type == "Incidents (Events)":
    horizon_h = 24
    n_links = st.slider("Number of links monitored", 1, 200, 30, step=1)
    base_incidents_per_h = st.slider("Base incidents per hour (corridor avg.)", 0.0, 10.0, 1.2, step=0.1)
elif dataset_type == "Accidents (Cases)":
    n = st.slider("Number of cases", 200, 100_000, 3000, step=100)
elif dataset_type == "Driver Behavior (Cross-Section)":
    n = st.slider("Number of drivers", 200, 100_000, 3000, step=100)

# ------------------------- Generators -------------------------
def fundamental_speed(density: np.ndarray, model: str, vf: float, kj: float) -> np.ndarray:
    density = np.asarray(density)
    density = _clip(density, 1e-6, max(kj - 1e-6, 1.0))
    if model == "Greenshields":
        # v = vf (1 - k/kj)
        v = vf * (1 - density / kj)
    elif model == "Greenberg":
        # v = vf * ln(kj/k)
        v = vf * np.log(kj / density)
    else:  # Underwood
        # v = vf * exp(-k/k0) ; calibrate k0 ~ kj/2
        k0 = kj / 2.0
        v = vf * np.exp(-density / k0)
    return _clip(v, 0.0, 140.0)

def generate_flow_static(n: int, vf: float, kj: float, model: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # density spread realistically across regimes
    k = rng.uniform(low=5, high=max(kj-5, 6), size=n)
    v = fundamental_speed(k, model, vf, kj) + rng.normal(0, 3, size=n)
    v = _clip(v, 0, 140)
    q = k * v
    occ = _clip(k / kj * 100.0, 0, 100)
    df = pd.DataFrame({
        "Section_ID": np.arange(1, n+1),
        "Density_veh_km": k,
        "Speed_kmh": v,
        "Flow_veh_h": q,
        "Occupancy_pct": occ
    })
    return df

def diurnal_profile(day_type: str) -> np.ndarray:
    """
    Normalized 24h shape (sum ~ 24) for demand. Double-peak weekday, flatter weekend.
    """
    hrs = np.arange(24)
    if day_type == "Weekday":
        am = np.exp(-0.5*((hrs-8)/2.2)**2)
        pm = np.exp(-0.5*((hrs-17)/2.5)**2)
        base = 0.6*am + 0.7*pm + 0.25  # baseline activity
    else:  # Weekend
        mid = np.exp(-0.5*((hrs-14)/3.5)**2)
        base = 0.8*mid + 0.35
    base = base / np.mean(base)  # normalize to mean 1.0
    return base

def generate_flow_timeseries(n_sections: int, vf: float, kj: float, model: str,
                             base_vph: float, day_type: str, weather_mult: float,
                             ts_noise: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shape = diurnal_profile(day_type)  # length 24, mean ~1
    hours = np.arange(24)
    records = []

    # section heterogeneity
    k_bias = rng.uniform(0.85, 1.15, size=n_sections)
    v_noise_scale = rng.uniform(2.0, 5.0, size=n_sections)

    # per-hour factors with weather and stochasticity
    hourly_demand = base_vph * shape * weather_mult
    # small multiplicative noise per hour
    hour_eps = rng.normal(0, ts_noise, size=24)
    hour_eps = np.exp(hour_eps)  # lognormal-like
    hourly_demand = hourly_demand * hour_eps

    for s in range(n_sections):
        # draw density around an operating point that moves with demand
        for h in hours:
            k_star = min(max(kj*0.15 + 0.001*hourly_demand[h], 5), kj-5)  # modest coupling
            k = rng.normal(k_star * k_bias[s], 8.0)
            k = _clip(k, 2.0, kj-2.0)
            v = fundamental_speed(np.array([k]), model, vf, kj)[0] + rng.normal(0, v_noise_scale[s])
            v = _clip(v, 0.0, 140.0)
            q = k * v
            occ = _clip(k / kj * 100.0, 0, 100)
            records.append((s+1, h, k, v, q, occ))
    df = pd.DataFrame(records, columns=["Section_ID","Hour","Density_veh_km","Speed_kmh","Flow_veh_h","Occupancy_pct"])
    return df

def generate_incidents(horizon_h: int, n_links: int, base_rate: float, weather_mult: float,
                       day_type: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Hourly base rates by day type (more during peaks on weekdays)
    shape = diurnal_profile(day_type)
    link_factor = rng.uniform(0.6, 1.4, size=n_links)

    records = []
    for link in range(n_links):
        rate_t = base_rate * shape * weather_mult * link_factor[link]
        # NHPP counts for each hour
        counts = _nhpp_thinning(rate_t, horizon_h, seed + link + 13)
        # Realize events and attributes
        for hr in range(horizon_h):
            c = counts[hr]
            if c == 0:
                continue
            # Attribute distributions conditional on weather/hour
            sev_p = np.array([0.68, 0.26, 0.06])  # Minor/Major/Fatal
            if weather_mult > 1.25:  # rainy/foggy
                sev_p = np.array([0.60, 0.30, 0.10])
            for _ in range(c):
                itype = rng.choice(["Crash","Breakdown","Obstacle","Construction"], p=[0.62,0.23,0.10,0.05])
                sev = rng.choice(["Minor","Major","Fatal"], p=sev_p)
                rt = rng.gamma(shape=2.5, scale=6.0)  # response
                ct = rng.gamma(shape=3.5, scale=9.0)  # clearance
                sev_mult = {"Minor":1.0,"Major":1.5,"Fatal":2.2}[sev]
                records.append((link+1, hr, itype, sev, weather, round(rt*sev_mult,1), round(ct*sev_mult+rt*0.3,1)))
    df = pd.DataFrame(records, columns=["Link_ID","Hour","Incident_Type","Severity","Weather","Response_min","Clearance_min"])
    if df.empty:
        # guarantee a valid frame
        df = pd.DataFrame(columns=["Link_ID","Hour","Incident_Type","Severity","Weather","Response_min","Clearance_min"])
    return df

def generate_accidents(n: int, seed: int, weather: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Gaussian copula for speed, age, reaction time (latent)
    means = np.array([70.0, 36.0, 1.25])         # speed (km/h), age (yrs), reaction (s)
    stds  = np.array([14.0, 10.0, 0.25])
    corr  = np.array([
        [ 1.00, -0.10,  0.25],   # faster drivers slightly younger, slower reaction (higher value)
        [-0.10,  1.00, -0.05],
        [ 0.25, -0.05,  1.00]
    ])
    X = _gaussian_copula(n, means, stds, corr, seed)
    speed = _clip(X[:,0], 0, 160)
    age   = _clip(X[:,1], 18, 85).astype(int)
    react = _clip(X[:,2], 0.5, 3.5)

    veh = rng.choice(["Car","Truck","Bus","Motorcycle"], size=n, p=[0.58,0.12,0.08,0.22])
    road = rng.choice(["Urban","Rural","Highway"], size=n, p=[0.55,0.30,0.15])
    light = rng.choice(["Daylight","Night","Twilight"], size=n, p=[0.62,0.30,0.08])
    wx = weather if rng.uniform()<0.65 else rng.choice(["Clear","Rainy","Foggy"], p=[0.7,0.25,0.05])

    # Injury probability driven by speed, lighting, weather
    base = 0.02 + 0.002*np.maximum(speed-50, 0)  # grows with overspeed
    base += np.where(light=="Night", 0.025, 0.0)
    base += np.where(np.array(wx)=="Rainy", 0.015, 0.0)
    base += np.where(np.array(wx)=="Foggy", 0.030, 0.0)
    base = _clip(base, 0, 0.6)

    # Map to categories with softmax-like allocation
    # p(Fatal) ≤ 5–8% typical upper cap; tuned by base
    p_fatal = _clip(0.03 + 0.07*base, 0.005, 0.12)
    p_ser   = _clip(0.12 + 0.40*base, 0.05, 0.45)
    p_minor = _clip(0.35 + 0.40*(1-base), 0.20, 0.60)
    # normalize remainder to "None"
    rem = 1 - (p_fatal + p_ser + p_minor)
    p_none = _clip(rem, 0.0, 1.0)

    probs = np.vstack([p_none, p_minor, p_ser, p_fatal]).T
    # ensure rows sum to ~1
    probs = probs / probs.sum(axis=1, keepdims=True)

    levels = ["None","Minor","Serious","Fatal"]
    idx = [rng.choice(4, p=probs[i]) for i in range(n)]
    injury = [levels[i] for i in idx]

    df = pd.DataFrame({
        "Vehicle_Type": veh,
        "Road_Type": road,
        "Lighting": light,
        "Weather": wx,
        "Speed_kmh": speed,
        "Driver_Age": age,
        "Reaction_Time_s": react,
        "Injury_Level": injury
    })
    return df

def generate_behavior(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Latent correlated attributes via Gaussian copula
    means = np.array([35.0, 8.0, 1.3, 5.0])    # age, experience, reaction, aggression
    stds  = np.array([12.0, 5.0, 0.25, 2.0])
    corr  = np.array([
        [ 1.00,  0.55, -0.25, -0.10],
        [ 0.55,  1.00, -0.15, -0.20],
        [-0.25, -0.15,  1.00,  0.10],
        [-0.10, -0.20,  0.10,  1.00]
    ])
    X = _gaussian_copula(n, means, stds, corr, seed+77)
    age = _clip(X[:,0], 18, 80).astype(int)
    expy = _clip(X[:,1], 0, 55)
    react = _clip(X[:,2], 0.5, 3.5)
    aggr  = _clip(X[:,3], 0, 10)

    # Derived behaviors
    phone = (rng.beta(2,5, size=n) + 0.03*(aggr-5) - 0.02*(expy-8))
    phone = _clip(phone, 0, 1)
    phone_flag = (rng.uniform(size=n) < phone).astype(int)

    viol_rate = np.exp(-1.1 + 0.12*aggr - 0.02*expy + 0.03*(1.6-react))
    viol = rng.poisson(lam=_clip(viol_rate, 0, 5))

    df = pd.DataFrame({
        "Driver_ID": np.arange(1, n+1),
        "Age": age,
        "Experience_Years": np.round(expy,1),
        "Reaction_Time_s": np.round(react,2),
        "Aggressiveness_0_10": np.round(aggr,2),
        "Phone_Use_While_Driving": np.where(phone_flag==1,"Yes","No"),
        "Violations_Last_Year": viol
    })
    return df

# ------------------------- Generate -------------------------
params_record: Dict[str, Any] = dict(
    preset=preset,
    fd_model=fd_model, vf=vf, kj=kj,
    seed=seed, dataset=dataset_type,
    weather=weather,
    day_type=day_type if "Flow (Time Series" in dataset_type or "Incidents" in dataset_type else None
)

if dataset_type == "Traffic Flow (Static)":
    df = generate_flow_static(n, vf, kj, fd_model, seed)
elif dataset_type == "Traffic Flow (Time Series, 24h)":
    df = generate_flow_timeseries(n_sections, vf, kj, fd_model, base_vph, day_type, weather_flow_multiplier, ts_noise, seed)
elif dataset_type == "Incidents (Events)":
    df = generate_incidents(horizon_h, n_links, base_incidents_per_h, weather_incident_multiplier, day_type, seed)
elif dataset_type == "Accidents (Cases)":
    df = generate_accidents(n, seed, weather)
else:  # Driver Behavior
    df = generate_behavior(n, seed)

# ------------------------- Display -------------------------
st.header(f"📊 {dataset_type}")
st.caption("Preview of the generated synthetic dataset.")
st.dataframe(df.head(20), use_container_width=True)

# ------------------------- Validation -------------------------
if show_validation:
    st.subheader("🔍 Validation Summary")
    with st.expander("Distribution & sanity checks", expanded=True):
        c1, c2, c3 = st.columns(3)
        n_rows, n_cols = df.shape
        with c1:
            st.write(f"Records: **{n_rows}**")
            st.write(f"Variables: **{n_cols}**")
        with c2:
            miss = df.isna().mean().mean()
            st.write(f"Missingness (overall): **{miss:.3f}**")
            st.write("All numeric variables clipped to physical bounds where applicable.")
        with c3:
            if "Flow" in df.columns or "Flow_veh_h" in df.columns:
                flow_col = "Flow" if "Flow" in df.columns else "Flow_veh_h"
                flow_nonneg = (df[flow_col] >= 0).mean()
                st.write(f"{flow_col} ≥ 0 share: **{flow_nonneg:.3f}**")

    # simple correlation snapshot (numeric only)
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] >= 2:
        corr = num.corr().round(3)
        st.write("**Numeric correlation matrix (sample):**")
        st.dataframe(corr)

# ------------------------- Visualization -------------------------
if show_plots:
    st.subheader("📈 Visualizations")
    if dataset_type.startswith("Traffic Flow"):
        if "Static" in dataset_type:
            fig, ax = plt.subplots()
            ax.scatter(df["Density_veh_km"], df["Speed_kmh"], alpha=0.4)
            ax.set_xlabel("Density (veh/km)")
            ax.set_ylabel("Speed (km/h)")
            ax.set_title("Speed–Density")
            st.image(_png_from_matplotlib(fig))
            plt.close(fig)

            fig2, ax2 = plt.subplots()
            ax2.scatter(df["Density_veh_km"], df["Flow_veh_h"], alpha=0.4)
            ax2.set_xlabel("Density (veh/km)")
            ax2.set_ylabel("Flow (veh/h)")
            ax2.set_title("Flow–Density")
            st.image(_png_from_matplotlib(fig2))
            plt.close(fig2)

        else:
            # time series: pick a few sections
            sections = df["Section_ID"].unique()
            pick = sections[:min(6, len(sections))]
            fig, ax = plt.subplots()
            for s in pick:
                kk = df[df["Section_ID"]==s].sort_values("Hour")
                ax.plot(kk["Hour"], kk["Flow_veh_h"], marker="o", alpha=0.7)
            ax.set_xlabel("Hour")
            ax.set_ylabel("Flow (veh/h)")
            ax.set_title("Hourly Flow (sample sections)")
            st.image(_png_from_matplotlib(fig))
            plt.close(fig)

    elif dataset_type == "Incidents (Events)":
        if not df.empty:
            fig, ax = plt.subplots()
            df["Incident_Type"].value_counts().plot(kind="bar", ax=ax)
            ax.set_title("Incident Type Frequency")
            st.image(_png_from_matplotlib(fig)); plt.close(fig)

            fig2, ax2 = plt.subplots()
            df.groupby("Hour").size().plot(kind="bar", ax=ax2)
            ax2.set_title("Incidents per Hour")
            st.image(_png_from_matplotlib(fig2)); plt.close(fig2)
        else:
            st.info("No incidents generated under current parameters (try increasing base rate or rainy/foggy conditions).")

    elif dataset_type == "Accidents (Cases)":
        fig, ax = plt.subplots()
        df["Injury_Level"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Injury Level Distribution")
        st.image(_png_from_matplotlib(fig)); plt.close(fig)

        fig2, ax2 = plt.subplots()
        ax2.hist(df["Speed_kmh"], bins=30)
        ax2.set_title("Speed Distribution")
        st.image(_png_from_matplotlib(fig2)); plt.close(fig2)

    else:  # Behavior
        fig, ax = plt.subplots()
        ax.hist(df["Aggressiveness_0_10"], bins=25)
        ax.set_title("Aggressiveness Score Distribution")
        st.image(_png_from_matplotlib(fig)); plt.close(fig)

        fig2, ax2 = plt.subplots()
        ax2.hist(df["Violations_Last_Year"], bins=15)
        ax2.set_title("Violations per Driver")
        st.image(_png_from_matplotlib(fig2)); plt.close(fig2)

# ------------------------- Downloads -------------------------
st.markdown("---")
st.subheader("⬇️ Download")

csv_bytes = df.to_csv(index=False).encode("utf-8")
meta_bytes = _pack_metadata(dataset_type, seed, params_record)

c1, c2 = st.columns(2)
with c1:
    _download_bytes(
        name=f"{dataset_type.replace(' ','_').replace('(','').replace(')','')}_synthetic.csv",
        content=csv_bytes,
        label="Download CSV",
        mime="text/csv"
    )
with c2:
    _download_bytes(
        name=f"{dataset_type.replace(' ','_').replace('(','').replace(')','')}_metadata.json",
        content=meta_bytes,
        label="Download Metadata (JSON)",
        mime="application/json"
    )

st.caption("STTDG+ v1.3 — © 2025. For academic and research use.")
