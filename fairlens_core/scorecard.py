"""Module for calculating fairness metrics (demographic parity, equalized odds, predictive parity, individual fairness).

This module implements 4 core fairness metrics to audit ML model predictions:
1. Demographic Parity: Do all demographic groups have equal approval rates?
2. Equalized Odds: Do all groups have equal TPR (true positive rate) and FPR (false positive rate)?
3. Predictive Parity: Do all groups have equal precision (justified approvals)?
4. Individual Fairness: Are similar individuals treated similarly?

Violations are flagged when disparity exceeds thresholds (typically 0.10).
Conflicts are detected when metrics disagree (e.g., parity passes but equalized odds fails).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score


def calculate_demographic_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    group_names: Optional[List[str]] = None,
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Calculate demographic parity metric: max disparity in approval rates across groups.
    
    Demographic parity is violated when different demographic groups have significantly
    different approval rates. E.g., if 51% of White applicants are approved but only 14%
    of Black applicants are approved, the parity gap is 0.37.
    
    Args:
        y_true: Ground truth labels (not used, but kept for API consistency).
        y_pred: Model predictions (0 or 1).
        sensitive_attr: Sensitive attribute array (e.g., race or gender).
        group_names: Optional list of unique group names for output. If None, auto-detected.
        threshold: Violation threshold (default 0.1 per NIST/EU AI Act).
    
    Returns:
        Dict with keys:
            - metric: float, max disparity in approval rates
            - violated: bool, whether metric > threshold
            - threshold: float, the violation threshold
            - group_rates: dict, approval rate per group
    """
    df = pd.DataFrame({"pred": y_pred, "attr": sensitive_attr})
    
    if group_names is None:
        group_names = sorted(df["attr"].unique())
    
    group_rates = {}
    for group in group_names:
        group_pred = df[df["attr"] == group]["pred"]
        rate = group_pred.mean() if len(group_pred) > 0 else 0.0
        group_rates[group] = float(rate)
    
    rates = list(group_rates.values())
    metric = float(max(rates) - min(rates)) if rates else 0.0
    violated = metric > threshold
    
    return {
        "metric": metric,
        "violated": violated,
        "threshold": threshold,
        "group_rates": group_rates,
    }


def calculate_equalized_odds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    group_names: Optional[List[str]] = None,
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Calculate equalized odds metric: max disparity in TPR and FPR across groups.
    
    Equalized odds requires that all groups have equal true positive rates (TPR)
    and equal false positive rates (FPR). This detects when the model makes
    different types of errors across groups.
    
    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Model predictions (0 or 1).
        sensitive_attr: Sensitive attribute array.
        group_names: Optional list of unique group names.
        threshold: Violation threshold (default 0.1).
    
    Returns:
        Dict with keys:
            - metric: float, max disparity in TPR or FPR
            - violated: bool, whether metric > threshold
            - threshold: float
            - tpr_by_group: dict, TPR per group
            - fpr_by_group: dict, FPR per group
            - tpr_disparity: float, max TPR disparity
            - fpr_disparity: float, max FPR disparity
    """
    df = pd.DataFrame({"true": y_true, "pred": y_pred, "attr": sensitive_attr})
    
    if group_names is None:
        group_names = sorted(df["attr"].unique())
    
    tpr_by_group = {}
    fpr_by_group = {}
    
    for group in group_names:
        group_df = df[df["attr"] == group]
        if len(group_df) == 0:
            tpr_by_group[group] = 0.0
            fpr_by_group[group] = 0.0
            continue
        
        # TPR = TP / (TP + FN) = P(pred=1 | true=1)
        positives = group_df[group_df["true"] == 1]
        tpr = (positives["pred"].sum() / len(positives)) if len(positives) > 0 else 0.0
        tpr_by_group[group] = float(tpr)
        
        # FPR = FP / (FP + TN) = P(pred=1 | true=0)
        negatives = group_df[group_df["true"] == 0]
        fpr = (negatives["pred"].sum() / len(negatives)) if len(negatives) > 0 else 0.0
        fpr_by_group[group] = float(fpr)
    
    tpr_vals = list(tpr_by_group.values())
    fpr_vals = list(fpr_by_group.values())
    
    tpr_disparity = float(max(tpr_vals) - min(tpr_vals)) if tpr_vals else 0.0
    fpr_disparity = float(max(fpr_vals) - min(fpr_vals)) if fpr_vals else 0.0
    
    metric = max(tpr_disparity, fpr_disparity)
    violated = metric > threshold
    
    return {
        "metric": metric,
        "violated": violated,
        "threshold": threshold,
        "tpr_by_group": tpr_by_group,
        "fpr_by_group": fpr_by_group,
        "tpr_disparity": tpr_disparity,
        "fpr_disparity": fpr_disparity,
    }


def calculate_predictive_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    group_names: Optional[List[str]] = None,
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Calculate predictive parity metric: max disparity in precision across groups.
    
    Predictive parity checks if approved applicants have similar approval justification
    across groups. Precision = TP / (TP + FP) = P(true=1 | pred=1).
    
    Args:
        y_true: Ground truth labels.
        y_pred: Model predictions.
        sensitive_attr: Sensitive attribute array.
        group_names: Optional list of unique group names.
        threshold: Violation threshold (default 0.1).
    
    Returns:
        Dict with keys:
            - metric: float, max precision disparity
            - violated: bool
            - threshold: float
            - precision_by_group: dict, precision per group
    """
    df = pd.DataFrame({"true": y_true, "pred": y_pred, "attr": sensitive_attr})
    
    if group_names is None:
        group_names = sorted(df["attr"].unique())
    
    precision_by_group = {}
    
    for group in group_names:
        group_df = df[df["attr"] == group]
        if len(group_df) == 0:
            precision_by_group[group] = 0.0
            continue
        
        # Precision = TP / (TP + FP) = P(true=1 | pred=1)
        approved = group_df[group_df["pred"] == 1]
        if len(approved) == 0:
            precision = 0.0
        else:
            precision = approved["true"].mean()
        precision_by_group[group] = float(precision)
    
    prec_vals = list(precision_by_group.values())
    metric = float(max(prec_vals) - min(prec_vals)) if prec_vals else 0.0
    violated = metric > threshold
    
    return {
        "metric": metric,
        "violated": violated,
        "threshold": threshold,
        "precision_by_group": precision_by_group,
    }


def calculate_individual_fairness(
    y_pred: np.ndarray,
    features_df: pd.DataFrame,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Calculate individual fairness: std dev of predictions for k-nearest neighbors.
    
    Individual fairness checks if similar individuals (by feature distance) receive
    similar predictions. High variance in predictions for similar applicants suggests
    individual unfairness (e.g., decisions are unstable/noisy).
    
    Args:
        y_pred: Model predictions (0 or 1).
        features_df: Numeric features DataFrame for computing similarity.
        k: Number of nearest neighbors to compare (default 5).
    
    Returns:
        Dict with keys:
            - metric: float, mean std dev across all samples
            - unstable: bool, whether metric > 0.3
            - high_variance_count: int, number of samples with std > 0.3
    """
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import pdist, squareform
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    
    # Compute pairwise distances
    distances = squareform(pdist(features_scaled, metric="euclidean"))
    
    variances = []
    high_variance_count = 0
    
    for i in range(len(y_pred)):
        # Get k nearest neighbors (including self)
        nearest_idx = np.argsort(distances[i])[:k + 1]
        neighbor_preds = y_pred[nearest_idx]
        variance = float(np.std(neighbor_preds))
        variances.append(variance)
        
        if variance > 0.3:
            high_variance_count += 1
    
    metric = float(np.mean(variances)) if variances else 0.0
    unstable = metric > 0.3 or high_variance_count > 0
    
    return {
        "metric": metric,
        "unstable": unstable,
        "high_variance_count": high_variance_count,
    }


def generate_scorecard(
    df: pd.DataFrame,
    sensitive_attrs: List[str] = None,
    model_col: str = "model_prediction",
    label_col: str = "loan_approved",
) -> Dict[str, Any]:
    """
    Generate comprehensive fairness scorecard for a model.
    
    This is the main function. It calculates all 4 fairness metrics for each
    protected attribute and detects conflicts (when metrics disagree).
    
    Args:
        df: DataFrame with predictions, labels, and sensitive attributes.
        sensitive_attrs: List of sensitive attribute column names (default ['gender', 'race']).
        model_col: Name of model prediction column (default 'model_prediction').
        label_col: Name of ground truth label column (default 'loan_approved').
    
    Returns:
        Dict structured as:
        {
            "gender": {
                "demographic_parity": {metric, violated, ...},
                "equalized_odds": {...},
                "predictive_parity": {...},
                "individual_fairness": {...},
                "conflicts": [...]
            },
            "race": {...}
        }
    """
    if sensitive_attrs is None:
        sensitive_attrs = ["gender", "race"]
    
    y_pred = df[model_col].values
    y_true = df[label_col].values
    
    scorecard = {}
    
    for attr in sensitive_attrs:
        if attr not in df.columns:
            continue
        
        sensitive_values = df[attr].values
        group_names = sorted(df[attr].unique())
        
        # Calculate all 4 metrics
        dem_parity = calculate_demographic_parity(y_true, y_pred, sensitive_values, group_names)
        eq_odds = calculate_equalized_odds(y_true, y_pred, sensitive_values, group_names)
        pred_parity = calculate_predictive_parity(y_true, y_pred, sensitive_values, group_names)
        ind_fairness = calculate_individual_fairness(y_pred, df.select_dtypes(include=[np.number]))
        
        # Detect conflicts
        conflicts = []
        
        # Conflict: demographic parity passes but equalized odds fails
        if not dem_parity["violated"] and eq_odds["violated"]:
            conflicts.append({
                "metric_a": "demographic_parity",
                "metric_b": "equalized_odds",
                "conflict": "approval_parity_but_unequal_errors",
                "explanation": (
                    f"Approval rates are equal across {attr} groups (parity passes), "
                    f"but error rates (TPR/FPR) are unequal (equalized odds fails). "
                    f"Model makes different types of errors for different groups."
                ),
            })
        
        # Conflict: demographic parity fails but equalized odds passes
        if dem_parity["violated"] and not eq_odds["violated"]:
            conflicts.append({
                "metric_a": "demographic_parity",
                "metric_b": "equalized_odds",
                "conflict": "unequal_approval_but_equal_errors",
                "explanation": (
                    f"Approval rates differ across {attr} groups (parity fails), "
                    f"but error rates are equal (equalized odds passes). "
                    f"This may indicate legitimate differences in qualifications."
                ),
            })
        
        scorecard[attr] = {
            "demographic_parity": dem_parity,
            "equalized_odds": eq_odds,
            "predictive_parity": pred_parity,
            "individual_fairness": ind_fairness,
            "conflicts": conflicts,
        }
    
    return scorecard
