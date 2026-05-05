"""Module for AI-powered bias explanations using Google Gemini API.

This module uses Gemini to generate human-friendly explanations of fairness
metrics, contextualize bias findings, and provide actionable remediation
recommendations. Integrates seamlessly with scorecard, intersectional, and
counterfactual modules.

Requires: GOOGLE_API_KEY environment variable (loaded from .env file)
Install: pip install google-genai python-dotenv
"""

from typing import Any, Dict, List, Optional
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will fall back to environment vars

try:
    import google.genai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def configure_gemini(api_key: Optional[str] = None) -> bool:
    """
    Configure Gemini API connection.
    
    The function tries to find the API key in this order:
    1. Explicit api_key parameter
    2. GOOGLE_API_KEY environment variable (.env file is auto-loaded)
    3. Returns False if no key is found
    
    Args:
        api_key: Google API key. If None, looks for GOOGLE_API_KEY environment variable or .env file.
    
    Returns:
        bool: True if configuration successful, False otherwise.
    
    Example:
        >>> configure_gemini()  # Will auto-load from .env file
        True
        
        >>> configure_gemini('your-api-key-here')  # Explicit key
        True
    """
    if not GEMINI_AVAILABLE:
        print("⚠️  google-genai not installed.")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            return True
        except Exception as e:
            print(f"❌ Failed to configure Gemini with provided key: {e}")
            return False
    else:
        # Try to get from environment (auto-loaded from .env)
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("❌ GOOGLE_API_KEY not found!")
            print("""
📝 Setup Instructions:

1. Create a .env file in the project root with:
   GOOGLE_API_KEY=your-api-key-here

2. Get your API key from: https://makersuite.google.com/app/apikey

3. The .env file is automatically loaded (never commit it to GitHub!)

4. Alternatively, set environment variable:
   
   Linux/Mac:
      export GOOGLE_API_KEY="your-api-key-here"
   
   Windows PowerShell:
      $env:GOOGLE_API_KEY="your-api-key-here"
   
   Windows CMD:
      set GOOGLE_API_KEY=your-api-key-here
            """)
            return False
        
        try:
            genai.configure(api_key=api_key)
            return True
        except Exception as e:
            print(f"❌ Failed to configure Gemini: {e}")
            return False


def explain_demographic_parity_violation(
    metric: float,
    group_rates: Dict[str, float],
    attribute_name: str = "demographic_group",
    context: str = "loan approval",
) -> str:
    """
    Generate AI explanation for demographic parity violation using Gemini.
    
    Args:
        metric: Demographic parity disparity metric (0-1)
        group_rates: Dict mapping group names to approval rates
        attribute_name: Name of the demographic attribute (e.g., 'race', 'gender')
        context: Context of the decision (e.g., 'loan approval', 'job hiring')
    
    Returns:
        str: Human-readable explanation of the violation and its severity
    
    Example:
        >>> group_rates = {'Black': 0.14, 'White': 0.51}
        >>> explain_demographic_parity_violation(0.37, group_rates, 'race', 'loan approval')
        "This model exhibits significant racial bias in loan approval decisions...
    """
    if not GEMINI_AVAILABLE:
        # Fallback explanation
        sorted_rates = sorted(group_rates.items(), key=lambda x: x[1], reverse=True)
        if sorted_rates:
            best_group, best_rate = sorted_rates[0]
            worst_group, worst_rate = sorted_rates[-1]
            return (
                f"The model shows a {metric:.1%} approval rate disparity across {attribute_name} groups. "
                f"{best_group} applicants are approved at {best_rate:.1%} while "
                f"{worst_group} applicants are approved at {worst_rate:.1%}. "
                f"This {metric:.0%} gap is {'severe' if metric > 0.3 else 'moderate'} and likely violates fairness standards."
            )
        return f"Demographic parity violation: {metric:.1%} disparity across {attribute_name}"
    
    prompt = f"""
You are a fairness auditor explaining bias findings to non-technical stakeholders.

Demographic Parity Violation Report:
- Metric: {metric:.1%} (disparity between groups)
- Attribute: {attribute_name}
- Context: {context}
- Group Approval Rates: {group_rates}

Generate a concise, non-technical explanation that:
1. Clearly states what the violation means
2. Explains the real-world impact on affected groups
3. Quantifies the severity (is {metric:.1%} severe? yes, it's 3-4x the acceptable threshold if > 0.30)
4. Avoids jargon but remains accurate

Keep it to 2-3 sentences, suitable for a compliance report.
"""
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt, timeout=5)
        return response.text.strip() if response else f"Demographic parity violation: {metric:.1%}"
    except Exception as e:
        # Fallback to basic explanation
        sorted_rates = sorted(group_rates.items(), key=lambda x: x[1], reverse=True)
        if sorted_rates:
            best_group, best_rate = sorted_rates[0]
            worst_group, worst_rate = sorted_rates[-1]
            return (
                f"The model shows a {metric:.1%} approval rate disparity across {attribute_name} groups. "
                f"{best_group} applicants are approved at {best_rate:.1%} while "
                f"{worst_group} applicants are approved at {worst_rate:.1%}."
            )
        return f"Demographic parity violation: {metric:.1%}"


def explain_intersectional_disparities(
    disparities: List[Dict[str, Any]],
    attribute_combo: str,
) -> str:
    """
    Generate AI explanation for intersectional bias patterns using Gemini.
    
    Args:
        disparities: List of subgroup dicts with 'subgroup_label', 'value', 'flagged'
        attribute_combo: Description of the demographic combination (e.g., 'race × gender')
    
    Returns:
        str: Explanation of intersectional bias patterns
    
    Example:
        >>> disparities = [
        ...     {'subgroup_label': 'Black+Female', 'value': 0.0},
        ...     {'subgroup_label': 'White+Male', 'value': 0.73}
        ... ]
        >>> explain_intersectional_disparities(disparities, 'race × gender')
        "Compound discrimination detected: Black women face..."
    """
    if not GEMINI_AVAILABLE:
        flagged_count = sum(1 for d in disparities if d.get('flagged'))
        return f"{flagged_count} intersectional disparities detected in {attribute_combo}"
    
    # Find worst and best performing subgroups
    sorted_disp = sorted(disparities, key=lambda x: x['value'])
    worst = sorted_disp[0] if sorted_disp else {}
    best = sorted_disp[-1] if sorted_disp else {}
    
    prompt = f"""
You are a fairness auditor explaining compound discrimination findings.

Intersectional Bias Report:
- Demographic Combination: {attribute_combo}
- Worst Performing Group: {worst.get('subgroup_label', 'unknown')} ({worst.get('value', 0):.1%} approval)
- Best Performing Group: {best.get('subgroup_label', 'unknown')} ({best.get('value', 0):.1%} approval)
- Total Subgroups Analyzed: {len(disparities)}
- Disparities Flagged: {sum(1 for d in disparities if d.get('flagged'))}

Generate a powerful but accurate explanation that:
1. Names the compound discrimination (e.g., "Black women face compound racial and gender discrimination")
2. Quantifies the gap
3. Explains why this matters (compound discrimination often overlooked in single-attribute analysis)
4. Notes the human impact

Keep it to 3-4 sentences, emotionally resonant but professionally appropriate.
"""
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating explanation: {e}"


def generate_remediation_recommendations(
    scorecard_results: Dict[str, Any],
    context: str = "loan approval model",
) -> Dict[str, List[str]]:
    """
    Generate AI-powered remediation recommendations using Gemini.
    
    Args:
        scorecard_results: Output from generate_scorecard()
        context: Context of the model (e.g., 'loan approval model', 'hiring system')
    
    Returns:
        Dict mapping attribute names to lists of recommendations
    
    Example:
        >>> scorecard = generate_scorecard(df)
        >>> recs = generate_remediation_recommendations(scorecard, 'loan approval')
        >>> print(recs['race'][0])
        "Implement fairness-aware preprocessing: use stratified sampling..."
    """
    if not GEMINI_AVAILABLE:
        return {"error": ["Gemini API not available"]}
    
    violations = []
    for attr, metrics in scorecard_results.items():
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict) and metric_data.get('violated'):
                violations.append(f"{attr} - {metric_name}: {metric_data.get('metric', 0):.2%}")
    
    if not violations:
        return {attr: ["No violations detected - model appears fair"] for attr in scorecard_results.keys()}
    
    prompt = f"""
You are a machine learning fairness engineer providing remediation guidance.

Context: {context}
Violations Detected:
{chr(10).join('- ' + v for v in violations)}

For each violation, provide 2-3 concrete, actionable remediation strategies:
1. Data-level fixes (preprocessing, reweighting)
2. Model-level fixes (fairness constraints, threshold adjustment)
3. Monitoring fixes (ongoing auditing, fairness metrics tracking)

Format as a JSON-like structure but respond in plain text. Be specific and technical.
Keep each recommendation to 1-2 sentences.
"""
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        # Parse response into structured format
        recommendations = {}
        for attr in scorecard_results.keys():
            recommendations[attr] = [
                line.strip() 
                for line in response.text.strip().split('\n') 
                if line.strip() and any(keyword in line.lower() for keyword in ['fix', 'implement', 'use', 'apply', 'consider'])
            ][:3]  # Take up to 3 recommendations per attribute
        
        return recommendations if any(recommendations.values()) else {
            "general": ["Implement fairness-aware model retraining", 
                       "Audit data collection process for bias",
                       "Consider fairness constraints in model optimization"]
        }
    except Exception as e:
        return {"error": [f"Error generating recommendations: {e}"]}


def summarize_audit_findings(
    scorecard: Dict[str, Any],
    intersectional: Dict[str, Any],
    counterfactual_sample: List[Dict[str, Any]],
) -> str:
    """
    Generate executive summary of bias audit using Gemini.
    
    Args:
        scorecard: Output from generate_scorecard()
        intersectional: Output from run_intersectional_audit()
        counterfactual_sample: Sample of outputs from run_counterfactual_analysis()
    
    Returns:
        str: Executive summary suitable for reports or executive presentations
    
    Example:
        >>> scorecard = generate_scorecard(df)
        >>> audit = run_intersectional_audit(df)
        >>> counterfactual_results = batch_counterfactual_analysis(model, df)
        >>> summary = summarize_audit_findings(scorecard, audit, counterfactual_results)
        >>> print(summary)
    """
    if not GEMINI_AVAILABLE:
        return "Gemini API not available for summary generation"
    
    # Extract key statistics
    violated_metrics = sum(
        1 for attr_metrics in scorecard.values()
        for metric_data in attr_metrics.values()
        if isinstance(metric_data, dict) and metric_data.get('violated')
    )
    
    unstable_count = sum(1 for cf in counterfactual_sample if cf.get('unstable'))
    
    prompt = f"""
You are writing an executive summary for a bias audit report.

Key Findings:
- Total Fairness Violations: {violated_metrics}
- Intersectional Disparities: {len(intersectional.get('2_way', {}))} combinations analyzed
- Unstable Counterfactuals: {unstable_count}/{len(counterfactual_sample)} applicants affected by protected attribute changes
- Scorecard Results: {scorecard}

Write a compelling 150-200 word executive summary that:
1. Opens with the severity assessment (e.g., "This model demonstrates significant bias across multiple dimensions")
2. Highlights the most severe finding
3. Notes the intersectional pattern if present
4. Concludes with urgency level (HIGH/MEDIUM/LOW)
5. Is suitable for presentation to C-level executives and regulators

Make it clear, data-driven, and actionable.
"""
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating summary: {e}"


def explain_counterfactual_instability(
    analysis: Dict[str, Any],
    model_name: str = "unknown",
) -> str:
    """
    Generate AI explanation for counterfactual instability findings.
    
    Args:
        analysis: Output from run_counterfactual_analysis()
        model_name: Name of the model being audited
    
    Returns:
        str: Explanation of what the instability means and why it matters
    
    Example:
        >>> result = run_counterfactual_analysis(model, row, df)
        >>> explanation = explain_counterfactual_instability(result)
        >>> print(explanation)
    """
    if not GEMINI_AVAILABLE:
        status = "unstable (biased)" if analysis.get('unstable') else "stable (fair)"
        return f"Counterfactual analysis shows {status} predictions"
    
    changed_attrs = list(analysis.get('counterfactuals', {}).keys())
    unstable = analysis.get('unstable', False)
    
    prompt = f"""
You are explaining what counterfactual instability means for AI bias.

Counterfactual Analysis:
- Model: {model_name}
- Original Prediction: {analysis.get('original_prediction')}
- Protected Attributes Tested: {', '.join(changed_attrs)}
- Predictions Changed: {unstable}
- Original Explanation: {analysis.get('explanation', '')}

Write a concise explanation (2-3 sentences) that:
1. Explains what the instability means in plain English
2. Describes the legal/ethical risk (protected attribute discrimination)
3. Distinguishes direct bias (attribute in features) from indirect (proxy features)

Make it suitable for compliance documentation.
"""
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating explanation: {e}"


def generate_fairness_report(
    scorecard: Dict[str, Any],
    intersectional: Dict[str, Any],
    counterfactual_sample: List[Dict[str, Any]],
    model_name: str = "Model Under Audit",
    dataset_name: str = "Unknown Dataset",
) -> Dict[str, Any]:
    """
    Generate comprehensive fairness audit report with Gemini-powered explanations.
    
    This is the main function that orchestrates all Gemini explanations and
    produces a report suitable for PDF generation or web display.
    
    Args:
        scorecard: Output from generate_scorecard()
        intersectional: Output from run_intersectional_audit()
        counterfactual_sample: Sample of counterfactual analysis results
        model_name: Name of the model being audited
        dataset_name: Name of the dataset
    
    Returns:
        Dict with keys: executive_summary, detailed_findings, recommendations, risk_assessment
    
    Example:
        >>> scorecard = generate_scorecard(df)
        >>> audit = run_intersectional_audit(df)
        >>> counterfactual = batch_counterfactual_analysis(model, df, n_samples=50)
        >>> report = generate_fairness_report(scorecard, audit, counterfactual)
        >>> print(report['executive_summary'])
    """
    if not GEMINI_AVAILABLE:
        return {"error": "Gemini API not configured"}
    
    report = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "executive_summary": summarize_audit_findings(scorecard, intersectional, counterfactual_sample),
        "fairness_violations": [],
        "intersectional_findings": [],
        "counterfactual_findings": [],
        "recommendations": generate_remediation_recommendations(scorecard),
        "risk_level": "UNKNOWN",
    }
    
    # Add detailed findings with explanations
    for attr, metrics in scorecard.items():
        if metrics.get('demographic_parity', {}).get('violated'):
            explanation = explain_demographic_parity_violation(
                metrics['demographic_parity']['metric'],
                metrics['demographic_parity']['group_rates'],
                attribute_name=attr
            )
            report["fairness_violations"].append({
                "attribute": attr,
                "metric": "demographic_parity",
                "severity": metrics['demographic_parity']['metric'],
                "explanation": explanation
            })
    
    # Assess risk level
    violated_count = len(report["fairness_violations"])
    unstable_count = sum(1 for cf in counterfactual_sample if cf.get('unstable'))
    
    if violated_count >= 2 or unstable_count > len(counterfactual_sample) * 0.3:
        report["risk_level"] = "HIGH"
    elif violated_count >= 1:
        report["risk_level"] = "MEDIUM"
    else:
        report["risk_level"] = "LOW"
    
    return report
