import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def correlation_selection(X, threshold=0.85):
    """
    Remove highly correlated features
    """
    corr_matrix = pd.DataFrame(X).corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    drop_features = [
        column for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]

    X_reduced = pd.DataFrame(X).drop(columns=drop_features)
    return X_reduced
def mutual_information_selection(X, y, top_k=10):
    """
    Select top features based on mutual information
    """
    mi_scores = mutual_info_classif(X, y)
    mi_series = pd.Series(mi_scores, index=X.columns)
    selected_features = mi_series.sort_values(ascending=False).head(top_k).index
    return X[selected_features]
