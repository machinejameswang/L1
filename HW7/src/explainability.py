import pandas as pd

def try_shap_summary(model, X_sample, output_path=None):
    try:
        import shap
        import matplotlib.pyplot as plt
        explainer = shap.Explainer(model, X_sample)
        values = explainer(X_sample)
        shap.plots.beeswarm(values, show=False)
        if output_path:
            plt.tight_layout()
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
        return values
    except Exception as e:
        print("SHAP skipped:", e)
        return None
