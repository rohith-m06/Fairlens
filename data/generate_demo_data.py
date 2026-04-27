"""
generate_demo_data.py
---------------------
Synthetic loan dataset for FairLens — a bias detection & auditing tool.

PURPOSE: This script creates a dataset with KNOWN, MEASURABLE demographic
disparities so that FairLens can detect, flag, and demonstrate remediation.
This is a test fixture, like a labeled benchmark for a fairness audit pipeline.

The bias is intentionally visible enough to be detectable but blended with
legitimate financial factors — mimicking how real historical bias operated.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. POPULATION
# ---------------------------------------------------------------------------
N = 1000

race_choices   = ["White", "Black", "Hispanic", "Asian"]
race_weights   = [0.45,    0.25,    0.20,       0.10]
gender_choices = ["Male", "Female"]
gender_weights = [0.55,   0.45]

race   = RNG.choice(race_choices,   size=N, p=race_weights)
gender = RNG.choice(gender_choices, size=N, p=gender_weights)

# ---------------------------------------------------------------------------
# 2. DEMOGRAPHICS & FINANCIAL FEATURES
# (income / credit distributions reflect documented historical wealth gaps;
#  these are the *inputs* FairLens should interrogate)
# ---------------------------------------------------------------------------
income_mean = {"White": 70_000, "Black": 48_000, "Hispanic": 44_000, "Asian": 67_000}
income_std  = 24_000

age = RNG.integers(18, 71, size=N).astype(float)

income = np.array([
    max(15_000, RNG.normal(income_mean[r], income_std))
    for r in race
])

# Credit score loosely follows income (a legitimate correlation)
credit_score = np.clip(
    300 + (income - 15_000) / (150_000 - 15_000) * 550
    + RNG.normal(0, 50, size=N),
    300, 850
)

years_employed = np.clip(RNG.normal(8, 5, size=N), 0, 30)

edu_probs = {
    # (High School, Bachelors, Masters, PhD)
    "White":    [0.20, 0.45, 0.25, 0.10],
    "Black":    [0.35, 0.40, 0.20, 0.05],
    "Hispanic": [0.40, 0.38, 0.17, 0.05],
    "Asian":    [0.10, 0.40, 0.35, 0.15],
}
edu_labels = ["High School", "Bachelors", "Masters", "PhD"]
education  = np.array([RNG.choice(edu_labels, p=edu_probs[r]) for r in race])

# ZIP code correlated with race (proxy variable — a key FairLens test case)
zip_map   = {"White": "10001", "Black": "10002", "Hispanic": "10003", "Asian": "10004"}
zip_noise = RNG.random(N)
zip_code  = np.where(
    zip_noise < 0.90,
    [zip_map[r] for r in race],   # 90% live in their "home" zip
    RNG.choice(["10001","10002","10003","10004","10005"], size=N)
)

loan_requested = RNG.integers(5_000, 50_001, size=N).astype(float)

# ---------------------------------------------------------------------------
# 3. APPROVAL LABELS  (the "biased historical decision")
# ---------------------------------------------------------------------------
# Base approval probability from LEGITIMATE factors only
def legitimate_score(inc, cs, ye, edu, age_):
    score = 0.0
    score += (inc - 15_000) / (150_000 - 15_000) * 0.30   # income
    score += (cs  - 300)    / (850 - 300)         * 0.35   # credit score
    score += min(ye, 20)    / 20                  * 0.15   # employment
    edu_bonus = {"High School": -0.05, "Bachelors": 0.0,
                 "Masters": 0.05, "PhD": 0.07}
    score += edu_bonus[edu]
    if 25 <= age_ <= 55:
        score += 0.05
    elif age_ < 25 or age_ > 65:
        score -= 0.08
    return score   # roughly 0–1 range

base_scores = np.array([
    legitimate_score(income[i], credit_score[i], years_employed[i],
                     education[i], age[i])
    for i in range(N)
])

# Bias adjustments (the unfair component FairLens should surface)
# These are the GROUND TRUTH biases embedded for detection testing.
bias_adj = np.zeros(N)

race_bias   = {"White": +0.12, "Black": -0.18, "Hispanic": -0.14, "Asian": -0.04}
gender_bias = {"Male":  +0.06, "Female": -0.06}

for i in range(N):
    bias_adj[i] += race_bias[race[i]]
    bias_adj[i] += gender_bias[gender[i]]
    # Intersectional effect: Black women face compounding penalty
    if race[i] == "Black" and gender[i] == "Female":
        bias_adj[i] -= 0.10
    # ZIP-code proxy effect (redlining pattern for FairLens to detect)
    if zip_code[i] == "10002":
        bias_adj[i] -= 0.06

# Final probability = legitimate signal + bias + small noise
final_prob = np.clip(base_scores + bias_adj + RNG.normal(0, 0.05, N), 0.05, 0.95)
loan_approved = (RNG.random(N) < final_prob).astype(int)

# ---------------------------------------------------------------------------
# 4. ASSEMBLE DATAFRAME
# ---------------------------------------------------------------------------
df = pd.DataFrame({
    "person_id":         np.arange(1, N + 1),
    "age":               age.astype(int),
    "gender":            gender,
    "race":              race,
    "zip_code":          zip_code,
    "income":            income.round(2),
    "credit_score":      credit_score.round(1),
    "years_employed":    years_employed.round(1),
    "education":         education,
    "loan_amount_requested": loan_requested.astype(int),
    "loan_approved":     loan_approved,
})

# ---------------------------------------------------------------------------
# 5. TRAIN BIASED MODEL  (LogisticRegression that learned from biased labels)
#    This is the model FairLens will AUDIT.
# ---------------------------------------------------------------------------
feature_cols = ["age", "income", "credit_score", "years_employed",
                "loan_amount_requested"]
# Encode categoricals
df_enc = pd.get_dummies(df[feature_cols + ["gender", "race", "education", "zip_code"]],
                        drop_first=False)

scaler = StandardScaler()
X = scaler.fit_transform(df_enc)
y = df["loan_approved"].values

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X, y)

df["model_probability"]  = model.predict_proba(X)[:, 1].round(4)
df["model_prediction"]   = model.predict(X)

# ---------------------------------------------------------------------------
# 6. BIAS SUMMARY REPORT  (ground truth for verifying FairLens output)
# ---------------------------------------------------------------------------
SEP = "=" * 60

print(f"\n{SEP}")
print("  FAIRLENS DEMO — BIAS GROUND TRUTH REPORT")
print(f"{SEP}\n")

overall = df["loan_approved"].mean()
print(f"Overall approval rate:  {overall:.1%}\n")

print("── Approval rate by RACE ──────────────────────────────────")
race_rates = df.groupby("race")["loan_approved"].mean().sort_values()
for r, rate in race_rates.items():
    bar = "█" * int(rate * 30)
    print(f"  {r:<12} {rate:.1%}  {bar}")

print("\n── Approval rate by GENDER ────────────────────────────────")
for g, rate in df.groupby("gender")["loan_approved"].mean().items():
    print(f"  {g:<8} {rate:.1%}")

print("\n── Approval rate by RACE × GENDER (intersectional) ────────")
cross = df.groupby(["race","gender"])["loan_approved"].mean().unstack()
print(cross.map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A").to_string())

print("\n── High-credit applicants denied (credit_score > 750) ─────")
hi_credit = df[df["credit_score"] > 750]
denied_hi  = hi_credit[hi_credit["loan_approved"] == 0]
print(f"  Total high-credit applicants: {len(hi_credit)}")
print(f"  Denied despite high credit:   {len(denied_hi)}")
print()
by_race_denied = denied_hi.groupby("race").size()
by_race_total  = hi_credit.groupby("race").size()
denial_rate    = (by_race_denied / by_race_total).fillna(0)
for r in race_choices:
    n_denied = int(by_race_denied.get(r, 0))
    n_total  = int(by_race_total.get(r, 0))
    rate     = denial_rate.get(r, 0.0)
    print(f"  {r:<12} {n_denied:>3}/{n_total:<4} denied  ({rate:.1%})")

print("\n── ZIP code ↔ Race correlation (proxy variable signal) ─────")
zip_num  = pd.Categorical(df["zip_code"]).codes
race_num = pd.Categorical(df["race"]).codes
corr     = np.corrcoef(zip_num, race_num)[0, 1]
print(f"  Pearson r(zip_code, race) = {corr:.3f}")
print(f"  → FairLens should flag zip_code as a racial proxy\n")

print("── Model accuracy & demographic parity gap ─────────────────")
overall_acc = (df["model_prediction"] == df["loan_approved"]).mean()
print(f"  Model accuracy: {overall_acc:.1%}")
model_race = df.groupby("race")["model_prediction"].mean()
max_gap    = model_race.max() - model_race.min()
print(f"  Demographic parity gap (max − min across races): {max_gap:.3f}")
print(f"  → NIST/EU AI Act threshold is typically 0.10; gap here = {max_gap:.3f}\n")

print(f"{SEP}")
print("  Files saved: demo_loans.csv  |  demo_loans_sample.csv")
print(f"{SEP}\n")

# ---------------------------------------------------------------------------
# 7. SAVE
# ---------------------------------------------------------------------------
df.to_csv("demo_loans.csv", index=False)
df.sample(100, random_state=42).reset_index(drop=True).to_csv(
    "demo_loans_sample.csv", index=False
)
