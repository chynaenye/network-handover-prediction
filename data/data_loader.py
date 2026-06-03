import kagglehub
import pandas as pd
import os

def load_data():
    path = kagglehub.dataset_download("meruvakodandasuraj/cellular-network-handover-prediction-dataset")
    df = pd.read_csv(os.path.join(path, "network_logs_1.csv"))
    return df

df = load_data()