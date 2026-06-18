from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def train_svm(X, y, kernel="rbf", C=1.0, gamma="scale", degree=3):
    """
    Train an SVM model using scikit-learn.
    Returns the trained pipeline and the training accuracy.
    """
    # It is good practice to scale data for SVM
    scaler = StandardScaler()
    
    # Setup SVC parameters
    svc = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma if kernel != "linear" else "scale",
        degree=degree if kernel == "poly" else 3,
        coef0=1.0 # Useful for poly/sigmoid
    )
    
    model = Pipeline([
        ("scaler", scaler),
        ("svc", svc)
    ])
    
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    return model, accuracy
