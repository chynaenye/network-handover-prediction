def feature_engineering(df):
    # Create new features based on existing ones
    df['RSRP_diff'] = df['RSRP'] - df['RSRP'].shift(1)
    df['RSRQ_diff'] = df['RSRQ'] - df['RSRQ'].shift(1)
    df['SINR_diff'] = df['SINR'] - df['SINR'].shift(1)
    df["velocity_diff"] = df["Velocity(km/h)"] - df["Velocity(km/h)"].shift(1)
    df['RSRP_rolling_mean'] = df['RSRP'].rolling(window=5).mean()
    return df

def target_variable(df):
    #Creating a target variable 'Handover' which indicates whether a handover occurred (1) or not (0) based on changes in the PCI value between consecutive records.
    df['Handover'] = (df['PCI'] != df['PCI'].shift(1)).astype(int)
    df.loc[0, 'Handover'] = 0 #First record cannot be a handover since there is no previous record to compare with, so we set it to 0.
    return df

