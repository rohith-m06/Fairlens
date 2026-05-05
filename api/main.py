from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime

# Add parent directory to path to import fairlens_core
sys.path.insert(0, str(Path(__file__).parent.parent))

from fairlens_core import scorecard, visualizations, pdf_generator

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for demo data
DEMO_DATA_CACHE = None


def load_demo_data():
    """Load and cache demo data"""
    global DEMO_DATA_CACHE
    if DEMO_DATA_CACHE is None:
        demo_path = Path(__file__).parent.parent / "data" / "demo_loans.csv"
        DEMO_DATA_CACHE = pd.read_csv(demo_path)
    return DEMO_DATA_CACHE.copy()


def run_fairness_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """Run complete fairness audit on a dataset"""
    
    # Normalize column names to lowercase and handle common variations
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    
    # Create a flexible column mapping for common variations
    column_replacements = {
        "loan_amount_requested": "approval_decision",
        "loan_request_amount": "approval_decision",
        "years_employed": "employment_years",
        "employment_years": "employment_years",
        "loan_approved": "approval_decision",
        "approval": "approval_decision",
        "approved": "approval_decision",
        "decision": "approval_decision",
    }
    
    # Map columns intelligently - handle case insensitivity
    df_renamed = df.copy()
    
    # Try to find and map approval/decision column
    approval_cols = ["loan_approved", "approval", "approved", "decision", "loan_approval", "approval_decision"]
    for col in df_renamed.columns:
        col_lower = col.lower()
        if col_lower in approval_cols:
            df_renamed["approval_decision"] = df_renamed[col]
            break
    
    # Try to find employment years column
    employment_cols = ["years_employed", "employment_years", "years_employment", "tenure"]
    for col in df_renamed.columns:
        col_lower = col.lower()
        if col_lower in employment_cols:
            df_renamed["employment_years"] = df_renamed[col]
            break
    
    # Try to find loan amount column  
    loan_cols = ["loan_amount_requested", "loan_amount", "loan_request", "requested_amount"]
    for col in df_renamed.columns:
        col_lower = col.lower()
        if col_lower in loan_cols:
            df_renamed["loan_amount"] = df_renamed[col]
            break
    
    # Check for required columns (case-insensitive)
    required_cols = ["credit_score", "age", "gender", "race", "approval_decision"]
    
    for col in required_cols:
        col_found = False
        for df_col in df_renamed.columns:
            if df_col.lower() == col.lower():
                if df_col != col:
                    df_renamed[col] = df_renamed[df_col]
                col_found = True
                break
        
        if not col_found:
            available_cols = ", ".join(df.columns.tolist())
            raise ValueError(f"Missing required column: '{col}'. Available columns: {available_cols}")
    
    # Prepare data
    y_true = df_renamed["approval_decision"].values.astype(int)
    y_pred = df_renamed["approval_decision"].values.astype(int)  # Use actual as prediction for audit
    
    # Run metrics for different protected attributes
    metrics = {}
    violations = []
    
    # Gender disparity
    try:
        gender_result = scorecard.calculate_demographic_parity(
            y_true, y_pred, df_renamed["gender"].values, threshold=0.15
        )
        gender_disparity = gender_result["metric"]
        metrics["gender_disparity"] = gender_disparity
        
        if gender_result["violated"]:
            violations.append({
                "attribute": "Gender",
                "metric": "Demographic Parity",
                "severity": gender_disparity,
                "explanation": f"Gender groups have {gender_disparity*100:.1f}% approval rate disparity. "
                               f"Group rates: {gender_result['group_rates']}"
            })
    except Exception as e:
        print(f"Gender analysis error: {e}")
        metrics["gender_disparity"] = 0.0
    
    # Race disparity
    try:
        race_result = scorecard.calculate_demographic_parity(
            y_true, y_pred, df_renamed["race"].values, threshold=0.15
        )
        race_disparity = race_result["metric"]
        metrics["race_disparity"] = race_disparity
        
        if race_result["violated"]:
            violations.append({
                "attribute": "Race",
                "metric": "Demographic Parity",
                "severity": race_disparity,
                "explanation": f"Racial groups have {race_disparity*100:.1f}% approval rate disparity. "
                               f"Group rates: {race_result['group_rates']}"
            })
    except Exception as e:
        print(f"Race analysis error: {e}")
        metrics["race_disparity"] = 0.0
    
    # Age disparity
    try:
        age_result = scorecard.calculate_demographic_parity(
            y_true, y_pred, df_renamed["age"].values, threshold=0.15
        )
        age_disparity = age_result["metric"]
        metrics["age_disparity"] = age_disparity
        
        if age_result["violated"]:
            violations.append({
                "attribute": "Age",
                "metric": "Demographic Parity",
                "severity": age_disparity,
                "explanation": f"Age groups have {age_disparity*100:.1f}% approval rate disparity."
            })
    except Exception as e:
        print(f"Age analysis error: {e}")
        metrics["age_disparity"] = 0.0
    
    # Determine overall risk level
    max_disparity = max(metrics.values()) if metrics else 0
    if max_disparity > 0.30:
        risk_level = "HIGH"
    elif max_disparity > 0.15:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    # Generate summary
    approval_rate = y_true.mean()
    summary = f"Analyzed {len(df)} loan applications. "
    summary += f"Overall approval rate: {approval_rate*100:.1f}%. "
    
    if violations:
        summary += f"Detected {len(violations)} fairness violations. "
        summary += "Review the Violations section for details."
    else:
        summary += "No major fairness violations detected across protected attributes."
    
    return {
        "records_analyzed": len(df),
        "attributes_analyzed": len([m for m in metrics if metrics[m] > 0]),
        "metrics": metrics,
        "violations": violations,
        "risk_level": risk_level,
        "summary": summary,
        "approval_rate": float(approval_rate),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    return {"status": "FairLens API Running"}


@app.get("/api/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/diag")
async def diag():
    return {
        "cwd": os.getcwd(),
        "files": os.listdir("."),
        "sys_path": sys.path[:3]
    }


@app.post("/api/audit/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and analyze a CSV file"""
    
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if df.empty:
            raise ValueError("CSV file is empty")
        
        # Run audit
        results = run_fairness_audit(df)
        return results
        
    except ValueError as e:
        error_msg = str(e)
        print(f"Validation error: {error_msg}")
        raise HTTPException(status_code=400, detail=f"Data validation error: {error_msg}")
    except Exception as e:
        error_msg = str(e)
        print(f"File analysis error: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Error analyzing file: {error_msg}")


@app.get("/api/audit/demo")
async def load_demo():
    """Load and analyze demo data"""
    
    try:
        df = load_demo_data()
        results = run_fairness_audit(df)
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading demo data: {str(e)}")


@app.post("/api/export/pdf")
async def export_pdf(data: Dict[str, Any]):
    """Generate and download PDF report"""
    
    try:
        # Create PDF
        pdf = pdf_generator.FairLensPDFReport()
        pdf.add_page()
        
        # Add title
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "FairLens Bias Detection Audit Report", 0, 1, "C")
        
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "C")
        pdf.ln(5)
        
        # Summary section
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Executive Summary", 0, 1)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 4, data.get("summary", "No summary available"))
        pdf.ln(3)
        
        # Metrics section
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Fairness Metrics", 0, 1)
        pdf.set_font("Helvetica", "", 9)
        
        for metric_name, metric_value in data.get("metrics", {}).items():
            metric_display = metric_name.replace("_", " ").title()
            pdf.cell(0, 5, f"{metric_display}: {metric_value*100:.1f}%", 0, 1)
        
        pdf.ln(3)
        
        # Violations section
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Violations Detected", 0, 1)
        pdf.set_font("Helvetica", "", 9)
        
        violations = data.get("violations", [])
        if violations:
            for v in violations:
                pdf.multi_cell(0, 4, 
                    f"• {v.get('attribute', 'Unknown')}: {v.get('explanation', 'No details')}")
        else:
            pdf.cell(0, 5, "No violations detected", 0, 1)
        
        # Return PDF as download
        pdf_bytes = pdf.output()
        
        return StreamingResponse(
            iter([pdf_bytes]),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=FairLens_Report.pdf"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")


@app.post("/api/export/csv")
async def export_csv(data: Dict[str, Any]):
    """Export audit results as CSV"""
    
    try:
        output = io.StringIO()
        
        # Write header
        output.write("FairLens Audit Report\n")
        output.write(f"Generated,{datetime.now().isoformat()}\n\n")
        
        # Write summary
        output.write("Executive Summary\n")
        output.write(data.get("summary", "No summary") + "\n\n")
        
        # Write metrics
        output.write("Fairness Metrics\n")
        output.write("Metric,Value\n")
        for metric_name, metric_value in data.get("metrics", {}).items():
            output.write(f"{metric_name},{metric_value*100:.1f}%\n")
        
        output.write("\n")
        
        # Write violations
        output.write("Violations Detected\n")
        output.write("Attribute,Metric,Severity\n")
        for v in data.get("violations", []):
            output.write(f"{v.get('attribute', 'Unknown')},{v.get('metric', 'N/A')},{v.get('severity', 0)*100:.1f}%\n")
        
        csv_bytes = output.getvalue().encode()
        
        return StreamingResponse(
            iter([csv_bytes]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=FairLens_Report.csv"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating CSV: {str(e)}")


@app.post("/api/visualize")
async def create_visualizations(data: Dict[str, Any]):
    """Generate interactive Plotly visualizations"""
    
    try:
        metrics = data.get("metrics", {})
        
        charts = {}
        errors = {}
        
        # Disparity chart
        try:
            charts["disparity_chart"] = visualizations.create_disparity_chart(metrics)
        except Exception as e:
            error_msg = str(e)
            print(f"Disparity chart error: {error_msg}")
            errors["disparity_chart"] = error_msg
        
        # Risk gauge
        try:
            charts["risk_gauge"] = visualizations.create_risk_gauge(data.get("risk_level", "LOW"))
        except Exception as e:
            error_msg = str(e)
            print(f"Risk gauge error: {error_msg}")
            errors["risk_gauge"] = error_msg
        
        # Comparison chart
        try:
            charts["comparison_chart"] = visualizations.create_metric_comparison_chart(metrics)
        except Exception as e:
            error_msg = str(e)
            print(f"Comparison chart error: {error_msg}")
            errors["comparison_chart"] = error_msg
        
        # Violation chart
        violations = data.get("violations", [])
        if violations:
            try:
                charts["violation_chart"] = visualizations.create_violation_distribution(violations)
            except Exception as e:
                error_msg = str(e)
                print(f"Violation chart error: {error_msg}")
                errors["violation_chart"] = error_msg
        
        return {
            "status": "success",
            "charts": charts,
            "errors": errors if errors else None
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"Visualization error: {error_msg}")
        return {"status": "error", "message": error_msg, "charts": {}}
