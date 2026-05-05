# FairLens: AI Bias Detection & Remediation Tool

**FairLens** is a comprehensive machine learning bias audit platform designed for organizations to detect, quantify, and remediate unfair discrimination in ML models before deployment.

## 🎯 Project Mission

Detect & fix algorithmic bias before it harms people. FairLens audits loan approval, hiring, credit, and other high-stakes models by scanning for disparate impact across demographics.

## 🏗️ Five-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: REMEDIATION ENGINE                                 │
│ Apply fixes (reweighting, threshold optimization)            │
│ Show accuracy vs fairness trade-off                          │
└─────────────────────────────────────────────────────────────┘
        ↑
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: COUNTERFACTUAL ENGINE                              │
│ "Would Sarah get the loan if she were White?"               │
│ Flip protected attributes, compare predictions               │
└─────────────────────────────────────────────────────────────┘
        ↑
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: INTERSECTIONAL AUDIT                               │
│ Check bias across race × gender × age combinations          │
│ Detect compounding discrimination                            │
└─────────────────────────────────────────────────────────────┘
        ↑
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: FAIRNESS SCORECARD                                 │
│ Calculate 4 fairness metrics per protected attribute         │
│ (demographic parity, equalized odds, etc.)                  │
└─────────────────────────────────────────────────────────────┘
        ↑
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: DATA AUTOPSY                                       │
│ Scan raw dataset for bias before training                   │
│ (representation gaps, label bias, proxy variables)          │
└─────────────────────────────────────────────────────────────┘
```

## 👥 Team Roles & File Ownership

| Person | Responsibility | Files |
|--------|---|---|
| **Person 1** (You) | Fairness metrics, intersectional audit, counterfactual engine | `fairlens_core/scorecard.py`, `intersectional.py`, `counterfactual.py` |
| **Person 2** | Data bias detection, remediation, FastAPI backend | `fairlens_audit/data_autopsy.py`, `remediation.py`, `api/main.py` |
| **Person 3** | Streamlit frontend (all 5 layers) | `streamlit_app/app.py`, `pages/*.py` |
| **Person 4** | Demo dataset, integration tests, PDF reporting | `data/generate_demo_data.py`, `tests/`, `reports/` |

## 🛠️ Tech Stack

- **Backend**: Python 3.9+, FastAPI, pandas, numpy, scikit-learn, fairlearn
- **Frontend**: Streamlit, Plotly
- **Testing**: pytest
- **Reporting**: fpdf2
- **Data**: Synthetic CSV (loan approval dataset with intentional bias)

## 🚀 Live Demo

The project is deployed on Vercel and can be accessed at:
**[FairLens Web Dashboard](https://fairlens-ckatbrhzy-mrohith676-9400s-projects.vercel.app)**

---

## 💻 Local Development & Setup

Follow these steps to run FairLens on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/rohith-m06/Fairlens.git
cd Fairlens
```

### 2. Set Up Environment Variables
Create a `.env` file in the root directory and add your Google API Key (needed for AI-powered insights):
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Install Dependencies
It is recommended to use a virtual environment:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 4. Run the Project

#### Option A: Run the Web Dashboard (FastAPI + HTML) - **RECOMMENDED**
This is the same version as the Vercel deployment. Best for interactive analysis.

**Terminal 1 - Start Backend API:**
```bash
cd api
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 - Start Frontend Server:**
```bash
cd public
python -m http.server 3000
```

Then open your browser at **http://127.0.0.1:3000**

The frontend will:
- Display the FairLens bias detection dashboard
- Allow you to upload CSV files or use demo data
- Show fairness metrics and visualizations
- Enable PDF report downloads

#### Option B: Run the Streamlit UI
This is the original hackathon frontend with more granular controls.
```bash
streamlit run streamlit_app/app.py
```

### 5. Using the Application

**On the Dashboard:**
1. Click **"Use Demo Data"** to analyze the sample loan dataset
2. Or upload your own CSV file with columns: `age`, `gender`, `race`, `credit_score`, `years_employed`, `loan_amount_requested`, `loan_approved`
3. View fairness metrics for each demographic group
4. Download PDF reports for stakeholder sharing

**Fairness Metrics Explained:**
- **Demographic Parity**: Do all groups get approved at similar rates?
- **Equalized Odds**: Do false positive/negative rates match across groups?
- **Predictive Parity**: Is prediction accuracy equal across groups?
- **Individual Fairness**: Are similar individuals treated similarly?

### 6. Generate Demo Data (Optional)
If you need fresh data for testing:
```bash
python data/generate_demo_data.py
```

---

## 🏗️ Five-Layer Architecture

**Bank's Loan Approval Model**
- **Surface Accuracy**: 85% overall
- **Hidden Bias**: 
  - Black applicants: 14% approval
  - White applicants: 51% approval
  - Black women under 30: 9% approval
  - White men over 40: 59% approval
  - Model violates equalized odds (0.529 gap vs NIST threshold 0.10)

**FairLens finds this**, explains it, and suggests fixes.

## 📝 Features

✅ Demographic parity detection  
✅ Equalized odds analysis  
✅ Intersectional bias detection  
✅ Counterfactual what-if analysis  
✅ Proxy variable detection (redlining patterns)  
✅ Bias remediation recommendations  
✅ Interactive Streamlit dashboard  
✅ PDF audit reports  

## 🏛️ Folder Structure

```
fairlens/
├── README.md                    # This file
├── CONTEXT.md                   # Full project context for future sessions
├── requirements.txt             # Pip packages
├── data/
│   ├── generate_demo_data.py   # Generates demo_loans.csv
│   ├── demo_loans.csv          # 1000 rows with intentional bias
│   └── demo_loans_sample.csv   # 100 rows for fast iteration
├── fairlens_core/              # PERSON 1 MODULES
│   ├── __init__.py
│   ├── scorecard.py            # 4 metrics, violation flags
│   ├── intersectional.py       # n-way demographic combos
│   └── counterfactual.py       # what-if attribute flips
├── fairlens_audit/             # PERSON 2 MODULES
│   ├── __init__.py
│   ├── data_autopsy.py         # Data bias detection
│   └── remediation.py          # Fix recommendations
├── streamlit_app/              # PERSON 3 FRONTEND
│   ├── app.py
│   └── pages/
│       ├── 01_data_autopsy.py
│       ├── 02_fairness_scorecard.py
│       ├── 03_intersectional_audit.py
│       ├── 04_counterfactual.py
│       └── 05_remediation.py
├── api/                        # PERSON 2 BACKEND
│   └── main.py
├── tests/                      # PERSON 4 TESTING
│   ├── test_scorecard.py
│   ├── test_intersectional.py
│   ├── test_counterfactual.py
│   └── test_integration.py
└── reports/                    # PERSON 4 REPORTING
    └── generate_report.py
```

---

**Built for the 24-hour hackathon** | Team of 4

For detailed project context, see [CONTEXT.md](CONTEXT.md).
