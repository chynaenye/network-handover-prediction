import numpy as np

def data_strip_and_conversion(df): 
    df['RSRP'] = df['RSRP'].str.replace(' dBm', '', regex=False)
    df['RSRP'] = df['RSRP'].astype(float)
    df['RSRQ'] = df['RSRQ'].str.replace(' dB', '', regex=False)
    df['RSRQ'] = df['RSRQ'].astype(float)
    df['SINR'] = df['SINR'].str.replace(' dB', '', regex=False)
    df['SINR'] = df['SINR'].astype(float)
    df['Velocity(km/h)'] = df['Velocity(km/h)'].str.replace(' km/h', '', regex=False)
    df['Velocity(km/h)'] = df['Velocity(km/h)'].astype(float)
    df["Downlink(Mbps)"] = df["Downlink(Mbps)"].str .replace(' Mbps', '', regex=False)
    df["Downlink(Mbps)"] = df["Downlink(Mbps)"].astype  (float)
    df["Uplink(Mbps)"] = df["Uplink(Mbps)"].str.replace(' Mbps', '', regex=False)
    df["Uplink(Mbps)"] = df["Uplink(Mbps)"].astype(float)       

    return df


def drop_unnecessary_columns(df):
    df = df.drop(columns=['DeviceID', 'deviceModel', 'deviceMake', "Network provi.", "NetworkType"], inplace=False, errors='ignore')

    return df


def validate_ranges(df):
    df.loc[(df['RSRP'] > -44) | (df['RSRP'] < -140), 'RSRP'] = np.nan
    df.loc[(df['RSRQ'] > -3) | (df['RSRQ'] < -20), 'RSRQ'] = np.nan
    return df.isna().sum().sort_values(ascending=False)


def SINR_outliers(df):
    df.loc[df['SINR'] == 2147483647, 'SINR'] = np.nan
    df['SINR'] = df['SINR'].fillna(df['SINR'].median())
    return df.isna().sum().sort_values(ascending=False)


def handling_missing_values(df):
    #Handling missing values in these columns by dropping them because handovers cannot be predicted without location and cell information.
    df = df.dropna(subset=['Latitude', 'Longitude', 'PCI'], inplace=True)
    df['Downlink(Mbps)'] = df['Downlink(Mbps)'].fillna(df['Downlink(Mbps)'].median())
    df['Uplink(Mbps)'] = df['Uplink(Mbps)'].fillna(df['Uplink(Mbps)'].median())
    return df.isna().sum().sort_values(ascending=False)