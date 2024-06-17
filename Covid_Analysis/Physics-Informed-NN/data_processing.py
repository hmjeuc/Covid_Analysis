import pandas as pd
import numpy as np

def load_data():
    """
    Load the COVID-19 data from CSV files.

    Returns:
        confirmed (pd.DataFrame): DataFrame of confirmed cases.
        deaths (pd.DataFrame): DataFrame of death cases.
        recovered (pd.DataFrame): DataFrame of recovered cases.
    """
    confirmed = pd.read_csv('Physics-Informed-NN/data_pinn/time_series_covid19_confirmed_global.csv')
    deaths = pd.read_csv('Physics-Informed-NN/data_pinn/time_series_covid19_deaths_global.csv')
    recovered = pd.read_csv('Physics-Informed-NN/data_pinn/time_series_covid19_recovered_global.csv')
    return confirmed, deaths, recovered

def check_missing_values(df, name):
    """
    Check and print missing values in the dataset.

    Args:
        df (pd.DataFrame): DataFrame to check for missing values.
        name (str): Name of the dataset for display purposes.
    """
    print(f"Missing values in {name} dataset:")
    print(df.isnull().sum())

def impute_missing_values(df):
    """
    Impute missing values in the dataset.

    Args:
        df (pd.DataFrame): DataFrame with missing values.

    Returns:
        pd.DataFrame: DataFrame with imputed missing values.
    """
    df['Province/State'] = df['Province/State'].fillna('Unknown')
    df['Lat'] = df.groupby('Country/Region')['Lat'].transform(lambda x: x.fillna(x.mean()))
    df['Long'] = df.groupby('Country/Region')['Long'].transform(lambda x: x.fillna(x.mean()))
    df['Lat'] = df['Lat'].fillna(df['Lat'].mean())
    df['Long'] = df['Long'].fillna(df['Long'].mean())
    return df

def preprocess_data(df, value_name):
    """
    Preprocess the data by melting it and converting dates.

    Args:
        df (pd.DataFrame): DataFrame to preprocess.
        value_name (str): Value name for melting.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df_melted = df.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"],
                        var_name="Date", value_name=value_name)
    df_melted["Date"] = pd.to_datetime(df_melted["Date"], format='%m/%d/%y')
    return df_melted

def normalize_data(confirmed, recovered, deaths):
    """
    Normalize the data using log transformation.

    Args:
        confirmed (np.array): Array of confirmed cases.
        recovered (np.array): Array of recovered cases.
        deaths (np.array): Array of death cases.

    Returns:
        tuple: Normalized confirmed, recovered, and death cases.
    """
    confirmed = np.log1p(np.maximum(0, confirmed))
    recovered = np.log1p(np.maximum(0, recovered))
    deaths = np.log1p(np.maximum(0, deaths))
    return confirmed, recovered, deaths

def main_preprocessing():
    """
    Main preprocessing function that loads, checks, imputes, and preprocesses the data.

    Returns:
        tuple: Arrays of time, confirmed, recovered, and death cases.
    """
    confirmed, deaths, recovered = load_data()

    check_missing_values(confirmed, "confirmed")
    check_missing_values(deaths, "deaths")
    check_missing_values(recovered, "recovered")

    confirmed = impute_missing_values(confirmed)
    deaths = impute_missing_values(deaths)
    recovered = impute_missing_values(recovered)

    confirmed_long = preprocess_data(confirmed, "Confirmed")
    deaths_long = preprocess_data(deaths, "Deaths")
    recovered_long = preprocess_data(recovered, "Recovered")

    data = confirmed_long.merge(deaths_long, on=["Province/State", "Country/Region", "Lat", "Long", "Date"])
    data = data.merge(recovered_long, on=["Province/State", "Country/Region", "Lat", "Long", "Date"])

    data['New_Confirmed'] = data.groupby(['Country/Region', 'Province/State'])['Confirmed'].diff().fillna(0)
    data['New_Deaths'] = data.groupby(['Country/Region', 'Province/State'])['Deaths'].diff().fillna(0)

    data['New_Confirmed_MA7'] = data.groupby(['Country/Region', 'Province/State'])['New_Confirmed'].transform(lambda x: x.rolling(7, 1).mean())
    data['New_Deaths_MA7'] = data.groupby(['Country/Region', 'Province/State'])['New_Deaths'].transform(lambda x: x.rolling(7, 1).mean())

    data['Date'] = pd.to_datetime(data['Date'])
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days

    time = data['Days'].values
    confirmed = data['Confirmed'].values
    recovered = data['Recovered'].values
    deaths = data['Deaths'].values

    confirmed, recovered, deaths = normalize_data(confirmed, recovered, deaths)

    return time, confirmed, recovered, deaths
