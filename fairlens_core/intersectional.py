"""Module for detecting bias across demographic combinations (race × gender, etc.) using 2-way and 3-way intersections.

Intersectional fairness recognizes that discrimination can compound when multiple
protected attributes combine. For example, Black women may face discrimination that
is distinct from discrimination against Black people generally or women generally.

This module generates all 2-way and 3-way demographic combinations and calculates
approval rate disparities within each combination. It returns data structured for
Plotly heatmaps in the Streamlit frontend.
"""

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def generate_interaction_groups(
    df: pd.DataFrame,
    attributes: List[str],
) -> List[Tuple[str, ...]]:
    """
    Generate all 2-way and 3-way demographic attribute combinations.
    
    Args:
        df: DataFrame (used only to validate that attributes exist).
        attributes: List of attribute names (e.g., ['gender', 'race', 'age_bucket']).
    
    Returns:
        List of tuples representing 2-way and 3-way combinations.
        E.g., [('gender', 'race'), ('gender', 'age_bucket'), ('race', 'age_bucket'),
               ('gender', 'race', 'age_bucket'), ...]
    """
    # Filter attributes that exist in the dataframe
    valid_attrs = [attr for attr in attributes if attr in df.columns]
    
    combos = []
    
    # Add all 2-way combinations
    combos.extend(combinations(valid_attrs, 2))
    
    # Add all 3-way combinations
    if len(valid_attrs) >= 3:
        combos.extend(combinations(valid_attrs, 3))
    
    return combos


def calculate_subgroup_approval(
    df: pd.DataFrame,
    combo: Tuple[str, ...],
    model_col: str = "model_prediction",
    min_sample_size: int = 10,
) -> List[Dict[str, Any]]:
    """
    Calculate approval rates for all subgroups in a demographic combination.
    
    For each unique combination of attribute values, calculate the mean approval rate
    (model prediction). Skip subgroups with fewer than min_sample_size samples.
    
    Args:
        df: DataFrame with predictions and demographic attributes.
        combo: Tuple of attribute names (e.g., ('gender', 'race')).
        model_col: Name of model prediction column.
        min_sample_size: Minimum samples to include a subgroup (default 10).
    
    Returns:
        List of dicts with keys:
            - subgroup_label: str, human-readable label (e.g., "Female+Black")
            - value: float, approval rate for this subgroup
            - count: int, sample size
            - flagged: bool, whether disparity from max is > 0.2
    """
    # Groupby all attributes in combo
    grouped = df.groupby(list(combo))[model_col].agg(["mean", "count"])
    grouped = grouped.reset_index()
    grouped.columns = list(combo) + ["approval_rate", "count"]
    
    # Filter by minimum sample size
    grouped = grouped[grouped["count"] >= min_sample_size].copy()
    
    # Create human-readable labels
    subgroup_data = []
    
    for idx, row in grouped.iterrows():
        label = "+".join([str(row[attr]) for attr in combo])
        subgroup_data.append({
            "subgroup_label": label,
            "combo": combo,
            "value": float(row["approval_rate"]),
            "count": int(row["count"]),
        })
    
    # Calculate max and min to determine if disparities are flagged
    if subgroup_data:
        approval_rates = [item["value"] for item in subgroup_data]
        max_rate = max(approval_rates)
        min_rate = min(approval_rates)
        disparity = max_rate - min_rate
        
        # Flag subgroups where disparity > 0.2 (20 percentage points)
        for item in subgroup_data:
            item["disparity"] = disparity
            item["flagged"] = disparity > 0.2
    
    return subgroup_data


def run_intersectional_audit(
    df: pd.DataFrame,
    sensitive_attrs: List[str] = None,
    model_col: str = "model_prediction",
    min_sample_size: int = 10,
) -> Dict[str, Any]:
    """
    Run full intersectional audit on a DataFrame.
    
    This is the main function. It generates all 2-way and 3-way demographic
    combinations and calculates approval disparities for each.
    
    Args:
        df: DataFrame with predictions and demographic attributes.
        sensitive_attrs: List of attribute names (default ['gender', 'race']).
        model_col: Name of model prediction column.
        min_sample_size: Minimum samples per subgroup to include.
    
    Returns:
        Dict structured as:
        {
            "2_way": {
                "('gender', 'race')": [list of subgroup dicts],
                "('gender', 'age_bucket')": [...],
                ...
            },
            "3_way": {
                "('gender', 'race', 'age_bucket')": [...],
                ...
            }
        }
    """
    if sensitive_attrs is None:
        sensitive_attrs = ["gender", "race"]
    
    combos = generate_interaction_groups(df, sensitive_attrs)
    
    # Separate 2-way and 3-way combos
    two_way = [c for c in combos if len(c) == 2]
    three_way = [c for c in combos if len(c) == 3]
    
    audit_result = {
        "2_way": {},
        "3_way": {},
    }
    
    # Process 2-way combinations
    for combo in two_way:
        subgroup_data = calculate_subgroup_approval(df, combo, model_col, min_sample_size)
        combo_label = str(combo)
        audit_result["2_way"][combo_label] = subgroup_data
    
    # Process 3-way combinations
    for combo in three_way:
        subgroup_data = calculate_subgroup_approval(df, combo, model_col, min_sample_size)
        combo_label = str(combo)
        audit_result["3_way"][combo_label] = subgroup_data
    
    return audit_result


def format_for_heatmap(
    audit_result: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Format intersectional audit results for Plotly heatmap visualization.
    
    Args:
        audit_result: Output from run_intersectional_audit().
    
    Returns:
        Dict with keys "2_way" and "3_way", each containing list of heatmap-ready dicts:
        {
            "2_way": [
                {"x_label": "Female", "y_label": "Black", "value": 0.09, "flagged": True},
                ...
            ],
            ...
        }
    """
    heatmap_data = {"2_way": [], "3_way": []}
    
    for way in ["2_way", "3_way"]:
        for combo_label, subgroups in audit_result[way].items():
            for subgroup in subgroups:
                # For 2-way: split label into x and y
                # For 3-way: use first two as x, y and combine third
                parts = subgroup["subgroup_label"].split("+")
                
                if way == "2_way" and len(parts) == 2:
                    x_label, y_label = parts[0], parts[1]
                else:
                    x_label = "+".join(parts[:-1]) if len(parts) > 1 else parts[0]
                    y_label = parts[-1] if len(parts) > 1 else "overall"
                
                heatmap_entry = {
                    "x_label": x_label,
                    "y_label": y_label,
                    "value": subgroup["value"],
                    "count": subgroup["count"],
                    "flagged": subgroup["flagged"],
                    "disparity": subgroup.get("disparity", 0.0),
                }
                
                heatmap_data[way].append(heatmap_entry)
    
    return heatmap_data
