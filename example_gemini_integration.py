"""
EXAMPLE: Using Gemini API with FairLens

This example shows how to:
1. Configure Gemini API
2. Run fairness audits with AI-powered explanations
3. Generate reports with executive summaries
4. Get remediation recommendations

To run:
    export GOOGLE_API_KEY="your-api-key-here"
    python example_gemini_integration.py
"""

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import FairLens modules (now with Gemini support)
from fairlens_core import (
    generate_scorecard,
    run_intersectional_audit,
    batch_counterfactual_analysis,
    configure_gemini,
    explain_demographic_parity_violation,
    explain_intersectional_disparities,
    generate_remediation_recommendations,
    summarize_audit_findings,
    generate_fairness_report,
)

print("\n" + "="*80)
print("FAIRLENS + GEMINI API INTEGRATION EXAMPLE")
print("="*80)

# Step 1: Configure Gemini
print("\n1️⃣  Configuring Gemini API...")
api_key = os.getenv('GOOGLE_API_KEY')

if not api_key:
    print("""
    ❌ GOOGLE_API_KEY not set!
    
    To use Gemini, set your API key:
    
    Linux/Mac:
        export GOOGLE_API_KEY="your-api-key-here"
    
    Windows (PowerShell):
        $env:GOOGLE_API_KEY="your-api-key-here"
    
    Windows (CMD):
        set GOOGLE_API_KEY=your-api-key-here
    
    Get your API key from: https://makersuite.google.com/app/apikey
    """)
    print("\n   Running without Gemini (explanations will be basic)...\n")
    gemini_ready = False
else:
    gemini_ready = configure_gemini(api_key)
    if gemini_ready:
        print("   ✅ Gemini API configured successfully")
    else:
        print("   ❌ Failed to configure Gemini")
        gemini_ready = False

# Step 2: Load data
print("\n2️⃣  Loading demo data...")
df = pd.read_csv('data/demo_loans.csv')
print(f"   ✅ Loaded {len(df)} rows")

# Step 3: Run basic fairness audit (without Gemini)
print("\n3️⃣  Running fairness audit...")

# Scorecard
print("   📊 Generating scorecard...")
scorecard = generate_scorecard(df, sensitive_attrs=['gender', 'race'])

# Intersectional
print("   📊 Running intersectional analysis...")
intersectional = run_intersectional_audit(df, sensitive_attrs=['gender', 'race'])

# Train model for counterfactual
print("   📊 Training model for counterfactual analysis...")
numeric_cols = ['age', 'income', 'credit_score', 'years_employed']
X = df[numeric_cols].fillna(df[numeric_cols].mean())
y = df['loan_approved']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_scaled, y)

# Counterfactual (smaller sample for faster execution)
print("   📊 Running counterfactual analysis (sample of 20 rows)...")

class ScaledModelWrapper:
    def __init__(self, model, scaler, numeric_cols):
        self.model = model
        self.scaler = scaler
        self.numeric_cols = numeric_cols
    
    def predict(self, X_df):
        X_numeric = X_df[self.numeric_cols].fillna(X_df[self.numeric_cols].mean())
        X_scaled = self.scaler.transform(X_numeric)
        return self.model.predict(X_scaled)

scaled_model = ScaledModelWrapper(model, scaler, numeric_cols)
counterfactual_sample = batch_counterfactual_analysis(
    scaled_model,
    df,
    feature_cols=numeric_cols,
    n_samples=20
)

print("   ✅ Audit complete")

# Step 4: Use Gemini for AI-powered explanations
if gemini_ready:
    print("\n4️⃣  Generating AI-powered explanations with Gemini...")
    
    # Explanation 1: Demographic Parity Violation
    print("\n   🤖 Demographic Parity Explanation:")
    race_parity = scorecard['race']['demographic_parity']
    explanation = explain_demographic_parity_violation(
        race_parity['metric'],
        race_parity['group_rates'],
        attribute_name='race',
        context='loan approval'
    )
    print(f"   {explanation}\n")
    
    # Explanation 2: Intersectional Disparities
    print("   🤖 Intersectional Disparities Explanation:")
    gender_race_combos = next(
        iter(intersectional['2_way'].values()), []
    )
    if gender_race_combos:
        explanation = explain_intersectional_disparities(
            gender_race_combos,
            'race × gender'
        )
        print(f"   {explanation}\n")
    
    # Explanation 3: Remediation Recommendations
    print("   🤖 Remediation Recommendations:")
    recommendations = generate_remediation_recommendations(
        scorecard,
        context='loan approval model'
    )
    for attr, recs in recommendations.items():
        if isinstance(recs, list) and recs and recs[0] != "No violations detected - model appears fair":
            print(f"\n   For {attr}:")
            for rec in recs[:2]:  # Show top 2 recommendations
                print(f"   • {rec}")
    
    # Explanation 4: Executive Summary
    print("\n   🤖 Executive Summary:")
    summary = summarize_audit_findings(scorecard, intersectional, counterfactual_sample)
    print(f"   {summary}\n")
    
    # Explanation 5: Full Report
    print("   🤖 Generating comprehensive report...")
    report = generate_fairness_report(
        scorecard,
        intersectional,
        counterfactual_sample,
        model_name="Loan Approval Model",
        dataset_name="demo_loans.csv"
    )
    
    print(f"\n   📄 Report Summary:")
    print(f"   - Risk Level: {report['risk_level']}")
    print(f"   - Fairness Violations: {len(report['fairness_violations'])}")
    print(f"   - Model: {report['model_name']}")
    print(f"   - Dataset: {report['dataset_name']}")

else:
    print("\n4️⃣  Skipping Gemini explanations (API not configured)")
    print("    To enable: export GOOGLE_API_KEY='your-key'")
    print("    Get key from: https://makersuite.google.com/app/apikey")

# Step 5: Display basic audit results
print("\n5️⃣  Audit Results Summary:")
print(f"\n   Race - Demographic Parity:")
race_parity = scorecard['race']['demographic_parity']
print(f"   • Metric: {race_parity['metric']:.1%}")
print(f"   • Violated: {race_parity['violated']}")
print(f"   • Group rates: {race_parity['group_rates']}")

print(f"\n   Gender - Demographic Parity:")
gender_parity = scorecard['gender']['demographic_parity']
print(f"   • Metric: {gender_parity['metric']:.1%}")
print(f"   • Violated: {gender_parity['violated']}")
print(f"   • Group rates: {gender_parity['group_rates']}")

print(f"\n   Counterfactual Analysis:")
unstable = sum(1 for cf in counterfactual_sample if cf['unstable'])
print(f"   • Samples Analyzed: {len(counterfactual_sample)}")
print(f"   • Unstable Predictions: {unstable}/{len(counterfactual_sample)}")

print("\n" + "="*80)
print("✅ INTEGRATION EXAMPLE COMPLETE")
print("="*80)
print("""
📚 Next Steps:

1. Set Gemini API Key:
   export GOOGLE_API_KEY="your-key-here"
   
2. Integrate with Streamlit (Person 3):
   st.write(explain_demographic_parity_violation(...))
   
3. Integrate with FastAPI (Person 2):
   @app.get("/api/explain/parity")
   def explain_parity():
       return explain_demographic_parity_violation(...)
   
4. Use for PDF Reports (Person 4):
   summary = summarize_audit_findings(...)
   # Add to PDF with fpdf2

🎯 Gemini adds value by:
   • Generating human-readable explanations
   • Contextualizing bias in business terms
   • Suggesting concrete fixes
   • Creating executive summaries
   • Making reports more compelling
""")
