# FairLens — Complete Project Context

## Project Overview

**FairLens** is an AI bias detection and remediation tool for a 24-hour hackathon. It audits machine learning models for unfair discrimination before deployment.

## Demo Scenario

A bank has a loan approval model:
- **Accuracy on surface**: 85%
- **Hidden discriminatory behavior**:
  - White applicants: 51% approval
  - Black applicants: 14% approval
  - Male applicants: 36% approval
  - Female applicants: 25% approval
  - Black women under 30: 9% approval
  - White men over 40: 59% approval
  - Model parity gap: 0.529 (NIST threshold is 0.10)
  - ZIP code correlation with race: r ≈ -0.62 (redlining proxy)

**FairLens detects all of this**, explains why, and provides remediation options.

## Tech Stack

**Backend**: Python 3.9+, FastAPI, pandas, numpy, scikit-learn, fairlearn, fpdf2  
**Frontend**: Streamlit, Plotly  
**Testing**: pytest  
**Data**: Synthetic CSV with 1000 rows (demo_loans.csv) + 100-row sample

## Team & Responsibilities (24 hours, 4 people)

### **Person 1 (YOUR ROLE)**
**Modules**: Fairness scorecard, intersectional audit, counterfactual engine

Build these 3 components:

1. **scorecard.py**: Calculate 4 fairness metrics
   - Demographic parity difference
   - Equalized odds difference
   - Predictive parity (precision per group)
   - Individual fairness (k-NN stability)
   - Return violation flags + group breakdowns
   - Detect metric conflicts (e.g., parity passes but equalized odds fails)

2. **intersectional.py**: Detect bias in demographic combinations
   - Accept list of protected attributes ['gender', 'race', 'age_bucket']
   - Use itertools.combinations for 2-way & 3-way combos
   - For each combo, groupby and calculate mean approval per subgroup
   - Flag if disparity > 20 percentage points
   - Return Plotly heatmap data structure

3. **counterfactual.py**: What-if analysis
   - Accept a trained sklearn model, one row, list of attributes to flip
   - For each attribute, get all unique values from training data
   - For each value, create modified row and run model.predict()
   - Flag as "unstable" if predictions differ from original
   - Explains whether model is sensitive to protected attributes

### **Person 2**
**Modules**: Data autopsy, remediation, FastAPI backend
- Scan raw dataset for representation gaps and label bias
- Detect proxy variables that encode protected attributes
- Apply reweighting and threshold optimization fixes
- Create FastAPI routes that wrap all 5 layers

### **Person 3**
**Modules**: Streamlit frontend (all pages)
- 5 pages (one per layer)
- Upload CSV, select model, run audit
- Display scorecards, heatmaps, counterfactual results
- Show remediation recommendations

### **Person 4**
**Modules**: Demo data, integration testing, PDF reporting
- Generate demo_loans.csv with intentional bias
- Write integration tests
- Generate PDF audit reports
- Demo script & pitch preparation

## Key Constraints

- ⏱️ **24-hour hackathon** — scope is tight
- 👥 **4 people** — parallel work required
- 📊 **Demo dataset exists** — demo_loans.csv (1000 rows) + demo_loans_sample.csv (100 rows)
- 🤖 **Model already trained** — LogisticRegression in demo dataset (model_prediction column)
- 📁 **Person 1 scope ONLY**: Do NOT touch frontend, FastAPI, remediation, data generation

## Bias Signals to Detect

Your 3 modules should detect these specific signals in demo_loans.csv:

| Signal | Values | Metric |
|--------|--------|--------|
| Racial disparity | White 51% vs Black 14% approval | Demographic parity |
| Gender gap | Male 36% vs Female 25% | Demographic parity |
| Intersectional | Black women 9% vs White men 59% | Intersectional audit |
| Model parity gap | 0.529 (NIST 0.10 threshold) | Equalized odds |
| ZIP code proxy | r = -0.62 with race | Counterfactual sensitivity |

## Dataset Schema

Columns in demo_loans.csv (1000 rows):
- `person_id`: Unique ID
- `age`: Integer 18–71
- `gender`: "Male" or "Female"
- `race`: "White", "Black", "Hispanic", "Asian"
- `zip_code`: "10001"–"10005" (correlated with race)
- `income`: $15k–$150k (varies by race)
- `credit_score`: 300–850 (correlated with income)
- `years_employed`: 0–30
- `education`: "High School", "Bachelors", "Masters", "PhD"
- `loan_amount_requested`: $5k–$50k
- `loan_approved`: 0 or 1 (ground truth — biased historical decisions)
- `model_probability`: Model probability from LogisticRegression
- `model_prediction`: 0 or 1 (the biased model to audit)

Your modules compare `model_prediction` vs `loan_approved`.

## Packages Required

```
fastapi==0.104.1
uvicorn==0.24.0
streamlit==1.28.0
pandas==2.1.1
numpy==1.26.0
scikit-learn==1.3.2
fairlearn==0.10.0
plotly==5.17.0
fpdf2==2.7.0
pytest==7.4.3
```

## Folder Structure

```
fairlens/
├── README.md                    # Project overview
├── CONTEXT.md                   # This file
├── requirements.txt             # Pip packages
├── data/
│   ├── generate_demo_data.py   # Generates demo_loans.csv (Person 4)
│   ├── demo_loans.csv          # 1000 rows, intentional bias
│   └── demo_loans_sample.csv   # 100 rows, for fast iteration
├── fairlens_core/              # PERSON 1 MODULES
│   ├── __init__.py
│   ├── scorecard.py            # 4 metrics, violation flags, conflicts
│   ├── intersectional.py       # n-way demographic combos
│   └── counterfactual.py       # what-if attribute flips
├── fairlens_audit/             # PERSON 2 MODULES
│   ├── __init__.py
│   ├── data_autopsy.py         # Data bias detection
│   └── remediation.py          # Fix recommendations
├── streamlit_app/              # PERSON 3 FRONTEND
│   ├── app.py                  # Main entry point
│   └── pages/
│       ├── 01_data_autopsy.py
│       ├── 02_fairness_scorecard.py
│       ├── 03_intersectional_audit.py
│       ├── 04_counterfactual.py
│       └── 05_remediation.py
├── api/                        # PERSON 2 BACKEND
│   └── main.py                 # FastAPI routes
├── tests/                      # PERSON 4 TESTING
│   ├── test_scorecard.py
│   ├── test_intersectional.py
│   ├── test_counterfactual.py
│   └── test_integration.py
└── reports/                    # PERSON 4 REPORTING
    └── generate_report.py      # PDF export
```

## Next Steps for Person 1

1. ✅ Read this CONTEXT.md + README.md
2. Create scorecard.py with 4 metric functions
3. Create intersectional.py with 2-way & 3-way combo detection
4. Create counterfactual.py with what-if analysis
5. Add type hints to ALL functions
6. Add docstrings to ALL functions
7. Test on demo_loans.csv
8. Handoff outputs to Person 2 (FastAPI) & Person 3 (Streamlit)

---

See README.md for project architecture & quick start.
