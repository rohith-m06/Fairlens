"""
PDF Report Generator for FairLens Audit Results
"""

from fpdf import FPDF
from datetime import datetime
import json
import re
from typing import Dict, Any, List


def strip_emojis(text: str) -> str:
    """Remove emoji and special characters from text"""
    if not text:
        return ""
    # Remove emoji and other problematic characters
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+", re.UNICODE)
    return emoji_pattern.sub(r'', text)


class FairLensPDFReport(FPDF):
    """Custom PDF report for fairness audits"""
    
    def __init__(self):
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297
        self.title = "FairLens Audit Report"
        
    def header(self):
        """Header with title and date"""
        self.set_font('Helvetica', 'B', 20)
        self.cell(0, 10, 'FairLens Bias Detection Report', ln=True, align='C')
        self.set_font('Helvetica', '', 10)
        self.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True, align='C')
        self.ln(5)
        
    def footer(self):
        """Footer with page number"""
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')
        
    def add_section(self, title: str):
        """Add section header"""
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(102, 126, 234)
        clean_title = strip_emojis(title)
        self.cell(0, 10, clean_title, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)
        
    def add_subsection(self, title: str):
        """Add subsection header"""
        self.set_font('Helvetica', 'B', 12)
        clean_title = strip_emojis(title)
        self.cell(0, 8, clean_title, ln=True)
        self.ln(1)
        
    def add_text(self, text: str, size: int = 10):
        """Add body text with wrapping"""
        self.set_font('Helvetica', '', size)
        clean_text = strip_emojis(text)
        self.multi_cell(0, 5, clean_text)
        self.ln(2)
        
    def add_metric_table(self, metrics: Dict[str, float]):
        """Add metrics table"""
        self.set_font('Helvetica', 'B', 10)
        self.set_fill_color(102, 126, 234)
        self.set_text_color(255, 255, 255)
        self.cell(100, 8, 'Metric', border=1, fill=True)
        self.cell(0, 8, 'Value', border=1, fill=True, ln=True)
        
        self.set_text_color(0, 0, 0)
        self.set_font('Helvetica', '', 9)
        
        for metric, value in metrics.items():
            clean_metric = strip_emojis(metric)
            self.cell(100, 7, clean_metric[:40], border=1)
            self.cell(0, 7, f'{value*100:.1f}%', border=1, ln=True)
            
        self.ln(3)
        
    def add_violations_section(self, violations: List[Dict[str, Any]]):
        """Add violations details"""
        if not violations:
            self.set_text_color(0, 128, 0)
            self.set_font('Helvetica', 'B', 11)
            self.add_text('No major fairness violations detected')
            self.set_text_color(0, 0, 0)
            return
            
        for i, violation in enumerate(violations, 1):
            self.set_font('Helvetica', 'B', 10)
            attr = violation.get('attribute', 'Unknown')
            severity = violation.get('severity', 0)
            self.add_text(f"Violation {i}: {attr} ({severity*100:.1f}%)")
            
            self.set_font('Helvetica', '', 9)
            explanation = violation.get('explanation', 'No details available')
            # Clean emojis from explanation
            explanation = strip_emojis(explanation)
            # Truncate to fit in PDF
            if len(explanation) > 300:
                explanation = explanation[:300] + '...'
            self.add_text(explanation)
            self.ln(1)


def generate_pdf_report(
    audit_results: Dict[str, Any],
    filename: str = "FairLens_Audit_Report.pdf"
) -> str:
    """
    Generate a PDF report from audit results
    
    Args:
        audit_results: Dictionary with metrics, violations, risk_level, summary
        filename: Output filename
        
    Returns:
        Path to generated PDF
    """
    
    pdf = FairLensPDFReport()
    pdf.add_page()
    
    # Executive Summary
    pdf.add_section("Executive Summary")
    summary = audit_results.get('summary', 'Fairness audit completed.')
    clean_summary = strip_emojis(summary)
    pdf.add_text(clean_summary)
    
    # Dataset Information
    pdf.add_section("Dataset Information")
    pdf.add_text(f"Records Analyzed: {audit_results.get('records_analyzed', 'N/A')}")
    pdf.add_text(f"Protected Attributes: {audit_results.get('attributes_analyzed', 'N/A')}")
    pdf.add_text(f"Risk Level: {audit_results.get('risk_level', 'Unknown').upper()}")
    
    # Fairness Metrics
    pdf.add_section("Fairness Metrics")
    metrics = audit_results.get('metrics', {})
    if metrics:
        pdf.add_metric_table(metrics)
    else:
        pdf.add_text("No metrics available")
    
    # Violations
    pdf.add_section("Violations Detected")
    violations = audit_results.get('violations', [])
    pdf.add_violations_section(violations)
    
    # Recommendations
    pdf.add_section("Recommendations")
    if violations:
        recommendations = """
1. Audit Training Data: Check for imbalanced representations
2. Review Feature Engineering: Identify proxy variables
3. Implement Fairness Constraints: Use fairlearn techniques
4. Rebalance Dataset: Ensure minority group representation
5. Monitor in Production: Continuous fairness tracking
        """.strip()
    else:
        recommendations = """
Your model demonstrates fair treatment across protected groups. Continue to:
- Monitor fairness metrics in production
- Regularly audit with new data
- Document fairness testing procedures
        """.strip()
    
    pdf.add_text(recommendations)
    
    # Compliance Notes
    pdf.add_section("Compliance Notes")
    pdf.add_text(
        "This report documents the fairness assessment performed on your model. "
        "Findings should be reviewed in context of applicable regulations "
        "(ECOA, FHA, GDPR, etc.). Consult legal team for compliance obligations."
    )
    
    # Save
    pdf.output(filename)
    return filename


if __name__ == "__main__":
    # Test
    test_results = {
        "summary": "Test audit completed",
        "records_analyzed": 1000,
        "attributes_analyzed": 2,
        "risk_level": "HIGH",
        "metrics": {"gender_disparity": 0.212, "race_disparity": 0.529},
        "violations": [
            {
                "attribute": "Gender",
                "severity": 0.212,
                "explanation": "Test violation"
            }
        ]
    }
    generate_pdf_report(test_results, "/tmp/test_report.pdf")
    print("PDF generated successfully")
