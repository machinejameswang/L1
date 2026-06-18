import numpy as np
import plotly.graph_objects as go

def plot_decision_boundary(X, y, model, kernel_name="rbf"):
    """
    Plot interactive decision boundary, support vectors, and data using Plotly.
    """
    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Control resolution based on dataset size for performance
    resolution = 100
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Get decision function scores
    Z = model.decision_function(grid)
    Z = Z.reshape(xx.shape)
    
    fig = go.Figure()

    # Decision surface contour
    fig.add_trace(
        go.Contour(
            x=xx[0], y=yy[:, 0], z=Z,
            colorscale=[[0, "#F43F5E"], [0.5, "#111827"], [1, "#38BDF8"]],
            opacity=0.6,
            showscale=False,
            contours=dict(showlines=False),
            hoverinfo="skip"
        )
    )
    
    # Decision boundary line (z=0)
    fig.add_trace(
        go.Contour(
            x=xx[0], y=yy[:, 0], z=Z,
            showscale=False,
            contours=dict(type="constraint", operation="=", value=0),
            line=dict(color="white", width=3),
            hoverinfo="skip",
            name="Decision Boundary"
        )
    )
    
    # Margin lines (z=-1, z=1)
    if kernel_name == "linear":
        for level in [-1, 1]:
            fig.add_trace(
                go.Contour(
                    x=xx[0], y=yy[:, 0], z=Z,
                    showscale=False,
                    contours=dict(type="constraint", operation="=", value=level),
                    line=dict(color="#A78BFA", width=2, dash="dash"),
                    hoverinfo="skip",
                    name=f"Margin {level}"
                )
            )

    # Data points
    colors = np.where(y == 0, "#38BDF8", "#F43F5E")
    fig.add_trace(
        go.Scatter(
            x=X[:, 0], y=X[:, 1],
            mode="markers",
            marker=dict(color=colors, size=8, line=dict(width=1, color="rgba(255,255,255,0.5)")),
            name="Data Points",
            hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>"
        )
    )
    
    # Support vectors
    svc = model.named_steps['svc']
    scaler = model.named_steps['scaler']
    sv_original = scaler.inverse_transform(svc.support_vectors_)
    
    fig.add_trace(
        go.Scatter(
            x=sv_original[:, 0], y=sv_original[:, 1],
            mode="markers",
            marker=dict(size=14, symbol="circle-open", color="#F59E0B", line=dict(width=3)),
            name="Support Vectors",
            hoverinfo="skip"
        )
    )
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False)
    )
    
    return fig
