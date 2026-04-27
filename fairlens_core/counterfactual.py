"""Module for counterfactual what-if analysis: flip protected attributes and detect if model predictions change.

Counterfactual fairness asks: "Would the outcome change if this person had a different
protected attribute value (race, gender, etc.)?" If yes, the model is directly or
indirectly using the protected attribute to make decisions.

This module supports:
- Flipping one or more protected attributes in a single applicant's data
- Re-running the model on modified versions
- Flagging instances where outcomes change (instability/sensitivity)
- Explaining whether the model is discriminatory at the individual level
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def get_unique_values(df: pd.DataFrame, attribute: str) -> List[Any]:
    """
    Get sorted unique values of an attribute from the DataFrame.
    
    Args:
        df: Training DataFrame.
        attribute: Column name.
    
    Returns:
        Sorted list of unique values, excluding NaN.
    """
    values = df[attribute].dropna().unique()
    return sorted([v for v in values if pd.notna(v)])


def create_counterfactual(
    row_dict: Dict[str, Any],
    attribute: str,
    new_value: Any,
) -> Dict[str, Any]:
    """
    Create a modified copy of a row with one attribute changed.
    
    Args:
        row_dict: Single row as a dictionary.
        attribute: Attribute name to flip.
        new_value: New value for the attribute.
    
    Returns:
        Modified copy of row_dict with attribute = new_value.
    """
    counterfactual_row = row_dict.copy()
    counterfactual_row[attribute] = new_value
    return counterfactual_row


def run_counterfactual_analysis(
    model: BaseEstimator,
    row_dict: Dict[str, Any],
    training_df: pd.DataFrame,
    flip_attributes: List[str] = None,
    feature_cols: List[str] = None,
) -> Dict[str, Any]:
    """
    Analyze how model predictions change when protected attributes are flipped.
    
    For each protected attribute, this function:
    1. Gets all unique values from the training set
    2. Creates modified versions of the row with each unique value
    3. Runs model.predict() on each modified version
    4. Compares predictions to detect instability
    
    Args:
        model: Fitted sklearn model with predict() method.
        row_dict: Single row as dictionary (keys must match training feature names).
        training_df: Full training DataFrame (to get unique attribute values).
        flip_attributes: List of attribute names to flip (default ['gender', 'race']).
        feature_cols: Optional list of column names the model was trained on.
                      If None, auto-detects numeric columns.
    
    Returns:
        Dict with keys:
            - original_prediction: int, model's prediction on original row
            - original_row: dict, the input row
            - counterfactuals: dict mapping attribute -> {value: prediction}
            - unstable: bool, whether any counterfactual differs from original
            - explanation: str, plain English explanation of findings
    """
    if flip_attributes is None:
        flip_attributes = ["gender", "race"]
    
    # Auto-detect feature columns if not provided
    if feature_cols is None:
        feature_cols = list(training_df.select_dtypes(include=[np.number]).columns)
    
    # Get original prediction (only pass feature columns to model)
    row_df = pd.DataFrame([row_dict])
    # Select only the feature columns that the model knows about
    if feature_cols:
        row_df = row_df[feature_cols]
    original_pred = int(model.predict(row_df)[0])
    
    counterfactuals = {}
    prediction_changed = False
    changed_attributes = []
    
    for attr in flip_attributes:
        if attr not in training_df.columns or attr not in row_dict:
            continue
        
        # Get all unique values for this attribute
        unique_vals = get_unique_values(training_df, attr)
        
        attr_results = {}
        
        for val in unique_vals:
            # Create counterfactual row
            cf_row = create_counterfactual(row_dict, attr, val)
            cf_df = pd.DataFrame([cf_row])
            # Select only the feature columns that the model knows about
            if feature_cols:
                cf_df = cf_df[feature_cols]
            
            # Get prediction
            cf_pred = int(model.predict(cf_df)[0])
            attr_results[val] = cf_pred
            
            # Check if prediction changed
            if cf_pred != original_pred:
                prediction_changed = True
                changed_attributes.append(attr)
        
        counterfactuals[attr] = attr_results
    
    # Generate explanation
    if prediction_changed:
        explanation = (
            f"Flipping {', '.join(set(changed_attributes))} changes the model's decision. "
            f"Original prediction: {original_pred}. "
            f"This indicates the model is sensitive to these protected attributes—either "
            f"directly (if they're input features) or indirectly (through proxy variables "
            f"like ZIP code that correlate with protected attributes)."
        )
    else:
        explanation = (
            "Flipping protected attributes does not change predictions. "
            "This person would receive the same decision regardless of their "
            f"{', '.join(flip_attributes)}. (Note: This is uncommon and may indicate "
            "the model has already 'learned' demographics indirectly.)"
        )
    
    return {
        "original_prediction": original_pred,
        "original_row": row_dict,
        "counterfactuals": counterfactuals,
        "unstable": prediction_changed,
        "explanation": explanation,
    }


def batch_counterfactual_analysis(
    model: BaseEstimator,
    df: pd.DataFrame,
    sample_indices: Optional[List[int]] = None,
    flip_attributes: List[str] = None,
    feature_cols: List[str] = None,
    n_samples: int = 10,
) -> List[Dict[str, Any]]:
    """
    Run counterfactual analysis on multiple rows.
    
    Args:
        model: Fitted sklearn model.
        df: Full DataFrame with predictions and features.
        sample_indices: Optional list of row indices to analyze. If None, sample randomly.
        flip_attributes: Attributes to flip (default ['gender', 'race']).
        feature_cols: Optional list of column names the model was trained on.
        n_samples: Number of rows to sample if sample_indices is None.
    
    Returns:
        List of counterfactual analysis results (one per row).
    """
    if flip_attributes is None:
        flip_attributes = ["gender", "race"]
    
    if feature_cols is None:
        feature_cols = list(df.select_dtypes(include=[np.number]).columns)
    
    if sample_indices is None:
        sample_indices = np.random.choice(len(df), min(n_samples, len(df)), replace=False)
    
    results = []
    for idx in sample_indices:
        row_dict = df.iloc[idx].to_dict()
        result = run_counterfactual_analysis(model, row_dict, df, flip_attributes, feature_cols)
        results.append(result)
    
    return results
