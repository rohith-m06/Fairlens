"""
Visualization Generator for FairLens - Creates interactive Plotly charts
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import json


def create_disparity_chart(metrics: Dict[str, float]) -> str:
    """Create bar chart showing fairness metric disparities"""
    
    names = [k.replace('_', ' ').title() for k in metrics.keys()]
    values = [v * 100 for v in metrics.values()]  # Convert to percentage
    colors = ['#e74c3c' if v > 30 else '#f39c12' if v > 15 else '#27ae60' for v in values]
    
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=values,
            marker=dict(color=colors),
            text=[f'{v:.1f}%' for v in values],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Disparity: %{y:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Fairness Metric Disparities',
        yaxis_title='Disparity (%)',
        xaxis_title='Protected Attributes',
        height=400,
        template='plotly_white',
        showlegend=False,
        hovermode='x unified'
    )
    
    # Add threshold lines
    fig.add_hline(y=15, line_dash='dash', line_color='orange', 
                  annotation_text='Warning Threshold (15%)', annotation_position='right')
    fig.add_hline(y=30, line_dash='dash', line_color='red',
                  annotation_text='Critical Threshold (30%)', annotation_position='right')
    
    return fig.to_json()


def create_risk_gauge(risk_level: str) -> str:
    """Create gauge chart for overall risk level"""
    
    risk_scores = {'LOW': 25, 'MEDIUM': 50, 'HIGH': 75}
    risk_colors = {'LOW': '#27ae60', 'MEDIUM': '#f39c12', 'HIGH': '#e74c3c'}
    
    score = risk_scores.get(risk_level, 0)
    color = risk_colors.get(risk_level, '#999')
    
    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=score,
        title={'text': 'Overall Fairness Risk Level'},
        delta={'reference': 50, 'suffix': ' vs. Medium'},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 33], 'color': '#d4edda'},
                {'range': [33, 66], 'color': '#fff3cd'},
                {'range': [66, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        font={'size': 12}
    )
    
    return fig.to_json()


def create_metric_comparison_chart(
    metrics: Dict[str, float],
    thresholds: Dict[str, float] = None
) -> str:
    """Create comparison chart with safe/warning/critical zones"""
    
    if thresholds is None:
        thresholds = {'safe': 0.15, 'warning': 0.30}
    
    metric_names = list(metrics.keys())
    metric_values = [v * 100 for v in metrics.values()]
    
    fig = go.Figure()
    
    # Add metric bars
    fig.add_trace(go.Bar(
        name='Actual Disparity',
        x=metric_names,
        y=metric_values,
        marker=dict(
            color=metric_values,
            colorscale=[[0, '#27ae60'], [0.5, '#f39c12'], [1, '#e74c3c']],
            cmin=0,
            cmax=50,
            showscale=True
        ),
        text=[f'{v:.1f}%' for v in metric_values],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Disparity: %{y:.1f}%<extra></extra>'
    ))
    
    # Add threshold reference lines
    fig.add_hline(y=15, line_dash='dash', line_color='orange', 
                  annotation_text='Safe Threshold', annotation_position='right')
    fig.add_hline(y=30, line_dash='dash', line_color='red',
                  annotation_text='Critical Threshold', annotation_position='right')
    
    fig.update_layout(
        title='Disparity Metrics vs. Safety Thresholds',
        yaxis_title='Disparity (%)',
        xaxis_title='Protected Attributes',
        height=450,
        template='plotly_white',
        showlegend=False,
        hovermode='x'
    )
    
    return fig.to_json()


def create_violation_distribution(violations: List[Dict[str, Any]]) -> str:
    """Create chart showing distribution of violations"""
    
    if not violations:
        # Empty state chart
        fig = go.Figure()
        fig.add_annotation(
            text='No violations detected',
            showarrow=False,
            font=dict(size=20, color='green')
        )
        fig.update_layout(title='Violation Analysis', height=300)
        return fig.to_json()
    
    attributes = [v.get('attribute', 'Unknown') for v in violations]
    severities = [v.get('severity', 0) * 100 for v in violations]
    
    # Assign distinct colors for each attribute
    color_map = {
        'Gender': '#FF6B6B',      # Bright red
        'Race': '#C92A2A',        # Dark red
        'Ethnicity': '#E03131',   # Medium red
        'Age': '#F76707',         # Orange
        'Male': '#4A90E2',        # Blue
        'Female': '#FF69B4',      # Hot pink
        'White': '#8B4513',       # Brown
        'Black': '#2C3E50',       # Dark gray
        'Asian': '#F39C12',       # Gold
        'Hispanic': '#D35400'     # Dark orange
    }
    
    # Get colors for each bar
    bar_colors = [color_map.get(attr, '#E74C3C') for attr in attributes]
    
    fig = go.Figure(data=[
        go.Bar(
            y=attributes,
            x=severities,
            orientation='h',
            marker=dict(
                color=bar_colors,
                line=dict(color='#333', width=1)
            ),
            text=[f'{s:.1f}%' for s in severities],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Severity: %{x:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Violation Severity by Attribute',
        xaxis_title='Severity (%)',
        yaxis_title='Protected Attribute',
        height=400,
        template='plotly_white',
        showlegend=False,
        font=dict(size=12)
    )
    
    return fig.to_json()


def create_dashboard(audit_results: Dict[str, Any]) -> Dict[str, str]:
    """Create complete dashboard with all visualizations"""
    
    return {
        'disparity_chart': create_disparity_chart(audit_results.get('metrics', {})),
        'risk_gauge': create_risk_gauge(audit_results.get('risk_level', 'UNKNOWN')),
        'comparison_chart': create_metric_comparison_chart(audit_results.get('metrics', {})),
        'violation_chart': create_violation_distribution(audit_results.get('violations', []))
    }


if __name__ == "__main__":
    # Test
    test_results = {
        "metrics": {"gender_disparity": 0.212, "race_disparity": 0.529},
        "risk_level": "HIGH",
        "violations": [
            {"attribute": "Gender", "severity": 0.212},
            {"attribute": "Race", "severity": 0.529}
        ]
    }
    
    dashboard = create_dashboard(test_results)
    print("Dashboard created successfully")
    print(f"Charts: {list(dashboard.keys())}")
