from posixpath import split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def train_test_split(df):
    X= df.drop(columns=['Timestamp', 'Handover']) #Dropping 'Timestamp' because it is not a useful feature for prediction and 'Handover' because it is the target variable.
    y = df['Handover']

    split.idx = int(len(df) * 0.8) #Using split because it is a time series data so as to preserve the temporal order and avoid data leakage.
    X_train = X.iloc[:split.idx]
    X_test = X.iloc[split.idx:]
    y_train = y.iloc[:split.idx]
    y_test = y.iloc[split.idx:]
    
    return X_train, X_test, y_train, y_test


def x_scaled_train_test_split(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def smote_train_test_split(X_train_scaled, y_train):
    smote = SMOTE(random_state=42) #Handling class imbalance in the training data using SMOTE (Synthetic Minority Over-sampling Technique) to create synthetic samples of the minority class, which can help improve the performance of the model on imbalanced datasets.
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    return X_train_smote, y_train_smote


def train_model_smote(X_train_smote, y_train_smote, model):
    model.fit(X_train_smote, y_train_smote)
    return model


def train_model(X_train_scaled, y_train, model):
    model.fit(X_train_scaled, y_train)
    return model


