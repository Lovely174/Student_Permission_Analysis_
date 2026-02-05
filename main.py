from data_loader import load_dataset
from preprocessing import (
    handle_missing_values,
    encode_categorical_features,
    normalize_features,
    create_target_variable
)
from feature_selection import correlation_selection, mutual_information_selection
from models import (
    train_logistic_regression,
    train_random_forest,
    train_svm
)
from evaluation import evaluate_model
from sklearn.model_selection import train_test_split

def main():
    # 1. Load dataset
    df = load_dataset(
        r"C:\Users\QORAISHA LOVELY\Downloads\student\student-mat.csv"
    )

    # 2. Preprocessing
    df = handle_missing_values(df)
    df = create_target_variable(df)
    df = encode_categorical_features(df)

    print("Preprocessing completed successfully")

    # 3. Define features and target  ✅ MUST COME BEFORE FEATURE SELECTION
    X = df.drop('performance', axis=1)
    y = df['performance']

    # 4. Hybrid Feature Selection
    X_corr = correlation_selection(X)
    X_selected = mutual_information_selection(X_corr, y, top_k=10)

    print("Hybrid feature selection completed")

    # 5. Normalize features
    X_scaled = normalize_features(X_selected)

    # 6. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 7. Train models
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    svm_model = train_svm(X_train, y_train)

    print("Model training completed successfully")

    # 8. Evaluate models
    evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    evaluate_model(rf_model, X_test, y_test, "Random Forest")
    evaluate_model(svm_model, X_test, y_test, "SVM")

if __name__ == "__main__":
    main()
