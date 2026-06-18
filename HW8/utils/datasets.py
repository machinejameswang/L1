import numpy as np
from sklearn.datasets import make_classification, make_blobs, make_moons, make_circles

def generate_dataset(dataset_name: str, n_samples: int = 300, noise: float = 0.1, random_state: int = 42):
    """
    Generate synthetic datasets for SVM visualization.
    """
    if dataset_name == "Linearly separable":
        X, y = make_classification(
            n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2,
            random_state=random_state, n_clusters_per_class=1, class_sep=2.0
        )
        # add some noise
        rng = np.random.RandomState(random_state)
        X += noise * rng.randn(n_samples, 2)
        
    elif dataset_name == "Blobs":
        X, y = make_blobs(
            n_samples=n_samples, centers=2, random_state=random_state, cluster_std=1.0 + noise*2
        )
        
    elif dataset_name == "Moons":
        X, y = make_moons(
            n_samples=n_samples, noise=noise, random_state=random_state
        )
        
    elif dataset_name == "Circles":
        X, y = make_circles(
            n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state
        )
        # Remap so inner is 0 (blue) and outer is 1 (red)
        y = 1 - y
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return X, y
