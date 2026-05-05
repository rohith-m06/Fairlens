"""
FairLens API - Vercel-Compatible Backend
Uses Person 1's fairlens_core modules for bias detection
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
import os
import sys
import csv
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add parent directory and current directory to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(Path(__file__).parent))

# Import Person 1's fairness modules
try:
    from fairlens_core import (
        generate_scorecard,
        run_intersectional_audit,
        explain_demographic_parity_violation,
        generate_fairness_report,
    )
    FAIRLENS_AVAILABLE = True
except ImportError as e:
    import traceback
    IMPORT_ERROR = f"Could not import fairlens_core: {e}\n{traceback.format_exc()}"
    print(IMPORT_ERROR)
    FAIRLENS_AVAILABLE = False

# Import visualization and PDF modules
try:
    from fairlens_core.visualizations import create_dashboard
    from fairlens_core.pdf_generator import generate_pdf_report
    EXTRAS_AVAILABLE = True
except ImportError as e:
    EXTRAS_AVAILABLE = False

# Initialize app
app = FastAPI(
    title="FairLens API",
    description="AI-Powered Bias Detection",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
public_dir = Path(__file__).parent.parent / "public"
if public_dir.exists():
    try:
        app.mount("/static", StaticFiles(directory=str(public_dir)), name="static")
    except:
        pass

@app.get("/")
async def root():
    """Serve homepage"""
    index = public_dir / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return {"status": "FairLens API running"}

@app.get("/api/health")
async def health():
    """Health check"""
    return {"status": "healthy", "service": "FairLens"}

@app.get("/api/audit/demo")
async def demo_audit():
    """Run audit on demo dataset"""
    try:
        # Find demo data
        paths = ["data/demo_loans.csv", "../data/demo_loans.csv", "demo_loans.csv"]
        df = None
        for p in paths:
            if os.path.exists(p):
                df = pd.read_csv(p)
                break
        
        if df is None:
            raise FileNotFoundError("Demo data not found")
        
        return analyze_dataframe(df)
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/audit/upload")
async def upload_audit(file: UploadFile = File(...)):
    """Upload and analyze CSV file"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="CSV file required")
        
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        
        if len(df) < 50:
            raise HTTPException(status_code=400, detail="Min 50 rows required")
        
        return analyze_dataframe(df)
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV")
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

def analyze_dataframe(df: pd.DataFrame) -> dict:
    """Run fairness audit on dataframe"""
    
    # Auto-detect columns (improved with more common names)
    pred_keywords = ['predict', 'pred', 'score', 'probability', 'prob', 'output']
    label_keywords = ['approved', 'label', 'target', 'y', 'outcome', 'default', 'ground_truth']
    sensitive_keywords = ['gender', 'race', 'ethnicity', 'age', 'sex', 'nationality', 'religion']
    
    pred_cols = [c for c in df.columns if any(k in c.lower() for k in pred_keywords)]
    label_cols = [c for c in df.columns if any(k in c.lower() for k in label_keywords)]
    
    # Prioritize 'prediction' over 'probability' for model_col if both exist
    if len(pred_cols) > 1:
        hard_preds = [c for c in pred_cols if 'prediction' in c.lower()]
        if hard_preds:
            model_col = hard_preds[0]
        else:
            model_col = pred_cols[0]
    else:
        model_col = pred_cols[0] if pred_cols else None
        
    label_col = label_cols[0] if label_cols else None
    
    if not model_col:
        raise ValueError(f"Could not find a prediction column. Found columns: {list(df.columns)}. Try naming your prediction column 'prediction' or 'score'.")
    if not label_col:
        raise ValueError(f"Could not find a label/target column. Found columns: {list(df.columns)}. Try naming your target column 'label' or 'approved'.")
    
    # Detect sensitive attributes
    attrs = []
    for col in df.columns:
        if any(x in col.lower() for x in sensitive_keywords):
            attrs.append(col)
    
    if not attrs:
        # Fallback if none detected, but warn or try most categorical-looking columns
        categorical_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].nunique() < 10]
        attrs = [c for c in categorical_cols if c not in [model_col, label_col]][:2]
    
    if not attrs:
        attrs = ['race', 'gender'] # Final fallback
    
    # Ensure they exist in DF
    attrs = [a for a in attrs if a in df.columns]
    
    metrics = {}
    violations = []
    
    if not FAIRLENS_AVAILABLE:
        raise HTTPException(status_code=500, detail=f"FairLens core modules not loaded. Error: {IMPORT_ERROR}")
    
    try:
        scorecard = generate_scorecard(
            df,
            sensitive_attrs=attrs,
            model_col=model_col,
            label_col=label_col
        )
        
        # Extract metrics
        for attr, attr_metrics in scorecard.items():
            if isinstance(attr_metrics, dict) and 'demographic_parity' in attr_metrics:
                parity = attr_metrics['demographic_parity']
                if isinstance(parity, dict):
                    val = float(parity.get('metric', 0))
                    metrics[f"{attr}_disparity"] = val
                    
                    if parity.get('violated'):
                        violations.append({
                            "attribute": attr.capitalize(),
                            "metric": "Demographic Parity",
                            "severity": val,
                            "explanation": explain_demographic_parity_violation(
                                metric=val,
                                group_rates=parity.get('group_rates', {}),
                                attribute_name=attr
                            )
                        })
    except Exception as e:
        print(f"Scorecard error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    # Risk level
    max_m = max(metrics.values()) if metrics else 0
    risk = "HIGH" if max_m > 0.3 else ("MEDIUM" if max_m > 0.15 else "LOW")
    
    return {
        "status": "success",
        "metrics": metrics,
        "violations": violations,
        "risk_level": risk,
        "summary": f"Analyzed {len(df)} records across {len(attrs)} protected attributes.",
        "records_analyzed": len(df),
        "attributes_analyzed": len(attrs)
    }

# Export & Visualization Endpoints

@app.post("/api/export/pdf")
async def export_pdf(audit_data: dict = Body(...)):
    """Generate PDF report from audit results"""
    try:
        if not EXTRAS_AVAILABLE:
            raise HTTPException(status_code=503, detail="PDF generation unavailable")
        
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            filepath = generate_pdf_report(audit_data, temp_path)
            
            # Read the file into memory
            with open(filepath, 'rb') as f:
                pdf_bytes = f.read()
            
            # Return as streaming response
            filename = f"FairLens_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            return StreamingResponse(
                BytesIO(pdf_bytes),
                media_type='application/pdf',
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    except Exception as e:
        print(f"PDF export error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

@app.post("/api/export/csv")
async def export_csv(audit_data: dict = Body(...)):
    """Export audit results as CSV"""
    try:
        # Use StringIO for CSV writing (csv module requires text, not bytes)
        output = StringIO()
        metrics = audit_data.get('metrics', {})
        violations = audit_data.get('violations', [])
        
        writer = csv.writer(output)
        
        # Metrics section
        writer.writerow(['FAIRNESS METRICS'])
        writer.writerow(['Metric', 'Value (%)'])
        for metric, value in metrics.items():
            writer.writerow([metric, f'{value*100:.2f}'])
        
        writer.writerow([])
        writer.writerow(['VIOLATIONS'])
        writer.writerow(['Attribute', 'Metric', 'Severity (%)', 'Explanation'])
        for v in violations:
            writer.writerow([
                v.get('attribute', ''),
                v.get('metric', ''),
                f'{v.get("severity", 0)*100:.2f}',
                v.get('explanation', '')[:100]
            ])
        
        writer.writerow([])
        writer.writerow(['SUMMARY'])
        writer.writerow(['Risk Level', audit_data.get('risk_level', 'N/A')])
        writer.writerow(['Records Analyzed', audit_data.get('records_analyzed', 0)])
        writer.writerow(['Attributes Analyzed', audit_data.get('attributes_analyzed', 0)])
        
        # Convert string to bytes
        csv_bytes = output.getvalue().encode('utf-8')
        
        # Return as streaming response
        filename = f"FairLens_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return StreamingResponse(
            BytesIO(csv_bytes),
            media_type='text/csv',
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        print(f"CSV export error: {e}")
        raise HTTPException(status_code=500, detail=f"CSV export failed: {str(e)}")

@app.post("/api/export/json")
async def export_json(audit_data: dict = Body(...)):
    """Export audit results as JSON"""
    try:
        json_data = json.dumps(audit_data, indent=2)
        
        filename = f"FairLens_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        return StreamingResponse(
            BytesIO(json_data.encode()),
            media_type='application/json',
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        print(f"JSON export error: {e}")
        raise HTTPException(status_code=500, detail=f"JSON export failed: {str(e)}")

@app.post("/api/visualize")
async def get_visualizations(audit_data: dict = Body(...)):
    """Generate visualization charts for audit results"""
    try:
        if not EXTRAS_AVAILABLE:
            return {
                "status": "unavailable",
                "message": "Visualizations not available"
            }
        
        dashboard = create_dashboard(audit_data)
        return {
            "status": "success",
            "charts": dashboard
        }
    except Exception as e:
        print(f"Visualization error: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
