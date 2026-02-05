import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def handle_missing_values(df):
    """
    Handle missing values if present
    """
    return df.dropna()

def encode_categorical_features(df):
    """
    Encode categorical features using Label Encoding
    """
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    return df

def normalize_features(X):
    """
    Normalize numerical features
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
def create_target_variable(df):
    """
    Convert final grade into binary target
    """
    df['performance'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
    df = df.drop(['G1', 'G2', 'G3'], axis=1)
    return df
