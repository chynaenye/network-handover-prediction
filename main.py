from data.data_loader import load_data
from src.data_preprocess import SINR_outliers, data_strip_and_conversion, drop_unnecessary_columns, validate_ranges, handling_missing_values
from src.train import train_test_split, x_scaled_train_test_split, smote_train_test_split, train_model_smote, train_model
from src.features import feature_engineering, target_variable
from src.evaluate import evaluate_model
from src.predict import predict_new

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def main():
    df = load_data()
    df = data_strip_and_conversion(df)
    df = drop_unnecessary_columns(df)
    df = validate_ranges(df)
    df = SINR_outliers(df)
    df = handling_missing_values(df)
    df = feature_engineering(df)
    df = target_variable(df)
    X_train, X_test, y_train, y_test = train_test_split(df)
    X_train_scaled, X_test_scaled, scaler = x_scaled_train_test_split(X_train, X_test)
    X_train_smote, y_train_smote = smote_train_test_split(X_train_scaled, y_train)
    
    # Train the model with SMOTE
    model_smote = train_model_smote(X_train_smote, y_train_smote, LogisticRegression(max_iter=1000)) #Replace 'model' with your chosen model.
    
    # Train the model without SMOTE
    model = train_model(X_train_scaled, y_train, RandomForestClassifier(class_weight='balanced', n_estimators=300)) #Replace 'model' with your chosen model.
    
    # Evaluate the model trained with SMOTE
    print("Evaluation of the model trained with SMOTE:")
    evaluate_model(model_smote, X_test_scaled, y_test)
    
    # Evaluate the model trained without SMOTE
    print("Evaluation of the model trained without SMOTE:")
    evaluate_model(model, X_test_scaled, y_test)

if __name__ == "__main__":
    main()
