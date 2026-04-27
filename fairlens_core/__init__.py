"""FairLens core: Fairness metrics, intersectional audit, counterfactual analysis, + Gemini-powered explanations.

Main modules:
- scorecard: Calculate 4 fairness metrics per demographic group
- intersectional: Detect bias across demographic combinations
- counterfactual: What-if analysis (flip protected attributes)
- gemini_explainer: AI-powered explanations using Google Gemini API
"""

from fairlens_core.counterfactual import (
    batch_counterfactual_analysis,
    create_counterfactual,
    get_unique_values,
    run_counterfactual_analysis,
)
from fairlens_core.gemini_explainer import (
    configure_gemini,
    explain_counterfactual_instability,
    explain_demographic_parity_violation,
    explain_intersectional_disparities,
    generate_fairness_report,
    generate_remediation_recommendations,
    summarize_audit_findings,
)
from fairlens_core.intersectional import (
    calculate_subgroup_approval,
    format_for_heatmap,
    generate_interaction_groups,
    run_intersectional_audit,
)
from fairlens_core.scorecard import (
    calculate_demographic_parity,
    calculate_equalized_odds,
    calculate_individual_fairness,
    calculate_predictive_parity,
    generate_scorecard,
)

__all__ = [
    # Scorecard
    "calculate_demographic_parity",
    "calculate_equalized_odds",
    "calculate_predictive_parity",
    "calculate_individual_fairness",
    "generate_scorecard",
    # Intersectional
    "generate_interaction_groups",
    "calculate_subgroup_approval",
    "run_intersectional_audit",
    "format_for_heatmap",
    # Counterfactual
    "get_unique_values",
    "create_counterfactual",
    "run_counterfactual_analysis",
    "batch_counterfactual_analysis",
    # Gemini Explainer
    "configure_gemini",
    "explain_demographic_parity_violation",
    "explain_intersectional_disparities",
    "generate_remediation_recommendations",
    "summarize_audit_findings",
    "explain_counterfactual_instability",
    "generate_fairness_report",
]

