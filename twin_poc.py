"""T2D Digital Twin PoC: fuses synthetic EHR (static) + synthetic CGM/wearable (dynamic)
to predict a hyperglycemic event (glucose > 180 mg/dL) within the next 2 hours.
All data is synthetic. Run: python twin_poc.py"""
import numpy as np, pandas as pd, json
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score
from sklearn.model_selection import GroupShuffleSplit

rng = np.random.default_rng(42)
N_PATIENTS, DAYS, STEP = 300, 14, 5          # 5-min samples
T = DAYS * 24 * 60 // STEP
HORIZON = 120 // STEP                          # 2 hours ahead
THRESH = 180

def make_ehr(n):
    return pd.DataFrame({
        "patient_id": np.arange(n),
        "age": rng.integers(30, 75, n),
        "bmi": rng.normal(28.5, 4.5, n).clip(18, 45),
        "hba1c": rng.normal(7.6, 1.2, n).clip(5.5, 12),
        "duration_yrs": rng.integers(0, 20, n),
        "family_history": rng.integers(0, 2, n),
        "on_metformin": rng.integers(0, 2, n),
    })

def simulate(p):
    t = np.arange(T); hour = (t * STEP / 60) % 24; day = t * STEP // 1440
    base = 28.7 * p.hba1c - 46.7                       # mean glucose from HbA1c
    resist = 0.6 + (p.bmi - 18) / 30 + 0.02 * p.duration_yrs - 0.15 * p.on_metformin
    glucose = np.full(T, base * 0.85)
    steps = np.zeros(T); stress = np.zeros(T)
    sleep = np.clip(rng.normal(6.5, 1.0, DAYS), 4, 9)
    for d in range(DAYS):
        for mh, carbs in [(8, rng.normal(60, 15)), (13.5, rng.normal(80, 20)), (20, rng.normal(70, 20))]:
            mt = int((d * 24 + mh + rng.normal(0, .4)) * 60 / STEP)
            k = np.arange(T) - mt
            m = k >= 0
            glucose[m] += resist * carbs * 0.55 * (k[m] / 9) * np.exp(1 - k[m] / 9) * (1.15 - 0.08 * sleep[d])
        for wh in [7, 18]:                              # walks lower glucose
            wt = int((d * 24 + wh) * 60 / STEP); steps[wt:wt+6] += rng.normal(600, 100, 6).clip(0)
            k = np.arange(T) - wt; m = (k >= 0) & (k < 36); glucose[m] -= 18 * np.sin(np.pi * k[m] / 36)
    steps += rng.poisson(15, T) * ((hour > 7) & (hour < 22))
    stress = np.convolve(rng.normal(0, 1, T), np.ones(24) / 24, "same") * 3
    glucose += stress * 3 + np.convolve(rng.normal(0, 4, T), np.ones(3) / 3, "same")
    glucose = glucose.clip(55, 400)
    hr = 68 + 0.04 * steps + 2 * stress + rng.normal(0, 2, T)
    hrv = (55 - 0.8 * stress - 0.1 * (p.age - 30) + rng.normal(0, 3, T)).clip(10)
    df = pd.DataFrame({"patient_id": p.patient_id, "t": t, "hour": hour, "glucose": glucose,
                       "hr": hr, "hrv": hrv, "steps": steps, "sleep_prev": sleep[day.astype(int)]})
    return df

ehr = make_ehr(N_PATIENTS)
data = pd.concat([simulate(r) for r in ehr.itertuples()], ignore_index=True)

def features(g):
    g = g.copy()
    for lag in (3, 6, 12):                              # 15/30/60 min deltas
        g[f"d_glu_{lag*5}"] = g.glucose.diff(lag)
    g["glu_mean_1h"] = g.glucose.rolling(12).mean()
    g["glu_max_1h"] = g.glucose.rolling(12).max()
    g["hr_mean_30"] = g.hr.rolling(6).mean()
    g["hrv_mean_1h"] = g.hrv.rolling(12).mean()
    g["steps_30"] = g.steps.rolling(6).sum()
    fut = g.glucose[::-1].rolling(HORIZON, min_periods=HORIZON).max()[::-1].shift(-1)
    g["label"] = (fut > THRESH).astype(float).where(fut.notna())
    return g

data = pd.concat([features(g) for _, g in data.groupby("patient_id")]).dropna()
data = data.merge(ehr, on="patient_id")
data = data[data.t % 3 == 0]                            # every 15 min

ehr_cols = ["age", "bmi", "hba1c", "duration_yrs", "family_history", "on_metformin"]
cgm_cols = ["glucose", "d_glu_15", "d_glu_30", "d_glu_60", "glu_mean_1h", "glu_max_1h"]
wear_cols = ["hr_mean_30", "hrv_mean_1h", "steps_30", "sleep_prev", "hour"]

tr_i, te_i = next(GroupShuffleSplit(test_size=0.25, random_state=0).split(data, groups=data.patient_id))
tr, te = data.iloc[tr_i], data.iloc[te_i]

def run(cols):
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0)
    m.fit(tr[cols], tr.label); p = m.predict_proba(te[cols])[:, 1]; yhat = p > 0.5
    return m, {"roc_auc": round(roc_auc_score(te.label, p), 3),
               "pr_auc": round(average_precision_score(te.label, p), 3),
               "recall": round(recall_score(te.label, yhat), 3),
               "precision": round(precision_score(te.label, yhat), 3)}

results = {"event_rate": round(float(te.label.mean()), 3), "n_patients": N_PATIENTS,
           "n_test_rows": int(len(te)),
           "glucose_only": run(cgm_cols)[1],
           "glucose+wearable": run(cgm_cols + wear_cols)[1]}
model, results["fused_ehr+cgm+wearable"] = run(ehr_cols + cgm_cols + wear_cols)
json.dump(results, open("results.json", "w"), indent=2)
print(json.dumps(results, indent=2))

# demo patient trace for dashboard
demo = te[te.patient_id == te.patient_id.iloc[0]].head(96).copy()
demo["risk"] = model.predict_proba(demo[ehr_cols + cgm_cols + wear_cols])[:, 1]
demo[["t", "glucose", "hr", "hrv", "steps", "risk"]].round(3).to_json("demo_patient.json", orient="records")
