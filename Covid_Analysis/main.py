import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objs as go
import plotly.subplots as sp
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from typing import Dict, List
from scipy.stats import ttest_ind

# URL Constants
DATA_URLS: Dict[str, str] = {
    'oxcgrt': 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_nat_latest.csv',
    'confirmed': 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
    'deaths': 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
    'recovered': 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv',
    'population': 'https://raw.githubusercontent.com/datasets/population/master/data/population.csv'
}

COUNTRIES_OF_INTEREST: List[str] = ['US', 'India', 'Germany', 'Brazil']
POLICIES_OF_INTEREST: List[str] = ['C1M_School closing', 'V4_Mandatory Vaccination (summary)', 'H6M_Facial Coverings']


def load_data(urls: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Load data from the provided URLs.

    Args:
        urls (Dict[str, str]): Dictionary of data source URLs.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of loaded data.
    """
    data = {name: pd.read_csv(url) for name, url in urls.items()}
    for name, df in data.items():
        if df.empty:
            raise ValueError(f"DataFrame {name} is empty.")
    return data

def preprocess_data(dfs: Dict[str, pd.DataFrame], countries: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Preprocess the data by filtering and transforming the input DataFrames.

    Args:
        dfs (Dict[str, pd.DataFrame]): A dictionary of DataFrames containing COVID-19 data.
        countries (List[str]): A list of country names to filter the data.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary of preprocessed DataFrames.
    """
    # Filter and reshape data
    dfs['confirmed'], dfs['deaths'], dfs['recovered'] = [
        df[df['Country/Region'].isin(countries)] for df in [dfs['confirmed'], dfs['deaths'], dfs['recovered']]
    ]
    dfs['confirmed'] = dfs['confirmed'].melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Confirmed')
    dfs['deaths'] = dfs['deaths'].melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Deaths')
    dfs['recovered'] = dfs['recovered'].melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name='Recovered')

    # Process OxCGRT data
    dfs['oxcgrt']['Date'] = pd.to_datetime(dfs['oxcgrt']['Date'], format='%Y%m%d')
    dfs['oxcgrt']['CountryName'] = dfs['oxcgrt']['CountryName'].replace({'United States': 'US'})
    dfs['oxcgrt'] = dfs['oxcgrt'][dfs['oxcgrt']['CountryName'].isin(countries)]

    # Process population data
    population_df = dfs['population']
    population_df = population_df[population_df['Year'] == population_df['Year'].max()]
    population_df = population_df[population_df['Country Name'].isin(countries)][['Country Name', 'Value']]
    population_df.columns = ['Country/Region', 'Population']
    dfs['population'] = population_df

    return dfs

def merge_data(dfs: Dict[str, pd.DataFrame], policy_columns: List[str]) -> pd.DataFrame:
    """
    Merge COVID-19 case data with policy data and population data.

    Args:
        dfs (Dict[str, pd.DataFrame]): A dictionary of DataFrames containing COVID-19 data.
        policy_columns (List[str]): A list of policy columns to include in the merged data.

    Returns:
        pd.DataFrame: A merged DataFrame containing the relevant data.
    """
    confirmed_df, deaths_df, recovered_df = dfs['confirmed'], dfs['deaths'], dfs['recovered']
    oxcgrt_df = dfs['oxcgrt']
    population_df = dfs['population']

    # Convert 'Date' columns to datetime to ensure consistency
    confirmed_df['Date'] = pd.to_datetime(confirmed_df['Date'], format='%m/%d/%y')
    deaths_df['Date'] = pd.to_datetime(deaths_df['Date'], format='%m/%d/%y')
    recovered_df['Date'] = pd.to_datetime(recovered_df['Date'], format='%m/%d/%y')
    oxcgrt_df['Date'] = pd.to_datetime(oxcgrt_df['Date'], format='%Y%m%d')

    # Select relevant columns from OxCGRT data
    oxcgrt_df = oxcgrt_df[['CountryName', 'Date'] + policy_columns]

    # Merge data
    merged_df = confirmed_df.merge(oxcgrt_df, left_on=['Country/Region', 'Date'], right_on=['CountryName', 'Date'], how='left') \
        .merge(deaths_df[['Country/Region', 'Date', 'Deaths']], on=['Country/Region', 'Date'], how='left') \
        .merge(recovered_df[['Country/Region', 'Date', 'Recovered']], on=['Country/Region', 'Date'], how='left') \
        .merge(population_df, on='Country/Region', how='left')
    merged_df.drop(['Province/State', 'Lat', 'Long', 'CountryName'], axis=1, inplace=True)

    # Handle missing values for case data only
    merged_df['Confirmed'].fillna(0, inplace=True)
    merged_df['Deaths'].fillna(0, inplace=True)
    merged_df['Recovered'].fillna(0, inplace=True)
    return merged_df

def add_features(df: pd.DataFrame, policy_columns: List[str], mean_infectious_period=5) -> pd.DataFrame:
    """
    Add features to the DataFrame for analysis.

    Args:
        df (pd.DataFrame): The merged DataFrame containing COVID-19 data and policy information.
        policy_columns (List[str]): A list of policy columns to analyze.
        mean_infectious_period (int): The average infectious period (default is 5 days).

    Returns:
        pd.DataFrame: The DataFrame with added features. The returned DataFrame contains the following columns:
            - 'DailyNewCases': The daily new cases for each country/region.
            - 'SmoothedDailyNewCases': The smoothed daily new cases for each country/region, calculated using a rolling window of 7 days.
            - 'DailyGrowthRate': The daily growth rate of cases for each country/region, calculated as the percentage change in smoothed daily new cases.
            - 'R0': The reproduction number (R0) for each country/region, calculated as 1 plus the daily growth rate divided by the mean infectious period.
            - For each policy column in `policy_columns`, the following columns are added:
                - '{policy_column}_Active': Indicates whether the policy is active (value > 0) or not.
                - '{policy_column}_Start': The date when the policy became active, or None if the policy is not active.
            - 'LaggedConfirmed': The number of confirmed cases 14 days prior for each country/region.
    """
    df['DailyNewCases'] = df.groupby('Country/Region')['Confirmed'].diff().fillna(0)
    df['SmoothedDailyNewCases'] = df.groupby('Country/Region')['DailyNewCases'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['DailyGrowthRate'] = df.groupby('Country/Region')['SmoothedDailyNewCases'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    df['R0'] = 1 + (df['DailyGrowthRate'] / mean_infectious_period)

    for policy_column in policy_columns:
        df[f'{policy_column}_Active'] = df[policy_column] > 0
        df[f'{policy_column}_Start'] = df.groupby('Country/Region')[f'{policy_column}_Active'].transform(lambda x: x.idxmax() if x.any() else None)

    df['LaggedConfirmed'] = df.groupby('Country/Region')['Confirmed'].shift(14)
    return df

def calculate_slope(df: pd.DataFrame, policy_column: str) -> Dict[str, float]:
    """
    Calculate the slope of the daily growth rate before and after policy implementation.

    Args:
        df (pd.DataFrame): DataFrame with additional features.
        policy_column (str): The policy column to analyze.

    Returns:
        Dict[str, float]: Dictionary with slopes for each country and policy status (True/False).
    """
    df['DaysSinceStart'] = (df['Date'] - df['Date'].min()).dt.days
    slopes = {}
    for country in df['Country/Region'].unique():
        for status in [True, False]:
            period_df = df[(df['Country/Region'] == country) & (df[f'{policy_column}_Active'] == status)]
            if len(period_df) > 1:
                X = period_df['DaysSinceStart'].values.reshape(-1, 1)
                y = period_df['DailyGrowthRate'].values
                slopes[(country, status)] = LinearRegression().fit(X, y).coef_[0]
    return slopes

def plot_trends(df: pd.DataFrame, country: str, policy_columns: List[str]) -> go.Figure:
    """
    Plot trends of confirmed cases, policy implementation, and R0 over time.

    Args:
        df (pd.DataFrame): DataFrame with additional features.
        country (str): Country to plot.
        policy_columns (List[str]): List of policy columns to analyze.

    Returns:
        go.Figure: Plotly figure object.
    """
    country_df = df[df['Country/Region'] == country]
    fig = sp.make_subplots(specs=[[{"secondary_y": True}]])

    # Plot confirmed cases
    fig.add_trace(go.Scatter(x=country_df['Date'], y=country_df['Confirmed'], name='Confirmed Cases', mode='lines'), secondary_y=False)

    # Plot policy implementation
    for policy_column in policy_columns:
        fig.add_trace(go.Scatter(x=country_df['Date'], y=country_df[policy_column], name=policy_column, mode='lines'), secondary_y=True)

    # Plot R0
    fig.add_trace(go.Scatter(x=country_df['Date'], y=country_df['R0'], name='R0', mode='lines'), secondary_y=True)

    fig.update_layout(title=f'Confirmed Cases, Policies, and R0 Over Time in {country}')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Confirmed Cases', secondary_y=False)
    fig.update_yaxes(title_text='Policy Implementation / R0', secondary_y=True)

    return fig

def prepare_data_for_prophet(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Prepare the data for Prophet modeling.

    Args:
        df (pd.DataFrame): DataFrame with additional features.
        country (str): Country to prepare data for.

    Returns:
        pd.DataFrame: DataFrame formatted for Prophet modeling.
    """
    prophet_df = df[df['Country/Region'] == country][['Date', 'R0']].rename(columns={'Date': 'ds', 'R0': 'y'}).dropna()
    return prophet_df

def fit_and_predict_prophet(prophet_df: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    """
    Fit the Prophet model and make predictions for the specified number of periods.

    Args:
        prophet_df (pd.DataFrame): DataFrame formatted for Prophet modeling.
        periods (int): Number of future periods to predict.

    Returns:
        pd.DataFrame: DataFrame with forecasted values.
    """
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

def plot_prophet_forecast(prophet_df: pd.DataFrame, forecast: pd.DataFrame, country: str) -> go.Figure:
    """
    Plot the Prophet forecast with annotations to explain each part of the plot.

    Args:
        prophet_df (pd.DataFrame): DataFrame formatted for Prophet modeling.
        forecast (pd.DataFrame): DataFrame with forecasted values.
        country (str): Country to plot.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = go.Figure()

    # Plot historical R0 values
    fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='markers', name='Observed R0'))

    # Plot forecasted R0 values
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted R0'))

    # Plot uncertainty intervals
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False, fill='tonexty', fillcolor='rgba(0, 176, 246, 0.2)', name='Uncertainty Interval'))

    fig.update_layout(
        title=f'R0 Prediction for {country}',
        xaxis_title='Date',
        yaxis_title='R0',
        legend_title='Legend'
    )

    return fig

def evaluate_model(forecast: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate the Prophet model by comparing the forecast with actual values.

    Args:
        forecast (pd.DataFrame): DataFrame with forecasted values.
        actual (pd.DataFrame): DataFrame with actual values.

    Returns:
        pd.DataFrame: DataFrame comparing forecasted and actual values.
    """
    forecast = forecast.set_index('ds')
    actual = actual.set_index('ds')
    comparison = forecast[['yhat', 'yhat_lower', 'yhat_upper']].join(actual, lsuffix='_forecast', rsuffix='_actual')
    comparison = comparison.dropna()

    mae = mean_absolute_error(comparison['y'], comparison['yhat'])
    mse = mean_squared_error(comparison['y'], comparison['yhat'])
    rmse = np.sqrt(mse)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    return comparison

def perform_t_test(df: pd.DataFrame, policy_column: str) -> Dict[str, Dict[str, float]]:
    """
    Perform t-test to compare daily growth rates before and after policy implementation.

    Args:
        df (pd.DataFrame): DataFrame with additional features.
        policy_column (str): The policy column to analyze.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary with t-test results for each country.
    """
    results = {}

    # Iterate through each unique country
    for country in df['Country/Region'].unique():
        # Find the index of the first active policy implementation date
        policy_start_idx = df[df[f'{policy_column}_Active'] & (df['Country/Region'] == country)].first_valid_index()
        
        if policy_start_idx is not None:
            policy_start_date = df.loc[policy_start_idx, 'Date']

            # Split the data into pre-policy and post-policy periods
            pre_policy_data = df[(df['Country/Region'] == country) & (df['Date'] < policy_start_date)]['DailyGrowthRate']
            post_policy_data = df[(df['Country/Region'] == country) & (df['Date'] >= policy_start_date)]['DailyGrowthRate']

            # Perform t-test if both pre-policy and post-policy data are sufficient
            if len(pre_policy_data) > 1 and len(post_policy_data) > 1:
                t_stat, p_value = ttest_ind(pre_policy_data, post_policy_data, equal_var=False)
                results[country] = {'t_stat': t_stat, 'p_value': p_value}
            else:
                results[country] = {'t_stat': np.nan, 'p_value': np.nan}
        else:
            results[country] = {'t_stat': np.nan, 'p_value': np.nan}

    return results

# Initialize the Dash app
app = dash.Dash(__name__)

# Load and preprocess data
dfs: Dict[str, pd.DataFrame] = load_data(DATA_URLS)
dfs = preprocess_data(dfs, COUNTRIES_OF_INTEREST)
merged_df: pd.DataFrame = merge_data(dfs, POLICIES_OF_INTEREST)

# Define layout
app.layout = html.Div([
    html.H1("COVID-19 Policy Impact Dashboard"),
    dcc.Dropdown(
        id='country-dropdown',
        options=[{'label': country, 'value': country} for country in COUNTRIES_OF_INTEREST],
        value='US'
    ),
    dcc.Dropdown(
        id='policy-dropdown',
        options=[{'label': policy, 'value': policy} for policy in POLICIES_OF_INTEREST],
        value='C1M_School closing'
    ),
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=merged_df['Date'].min(),
        end_date=merged_df['Date'].max(),
        display_format='YYYY-MM-DD'
    ),
    dcc.Graph(id='trends-graph'),
    dcc.Graph(id='prophet-forecast-graph'),
    html.Div(id='t-test-results')
])

# Define callbacks
@app.callback(
    [Output('trends-graph', 'figure'),
     Output('prophet-forecast-graph', 'figure'),
     Output('t-test-results', 'children')],
    [Input('country-dropdown', 'value'),
     Input('policy-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_graphs(selected_country: str, selected_policy: str, start_date: str, end_date: str):
    filtered_df = merged_df[(merged_df['Date'] >= start_date) & (merged_df['Date'] <= end_date)].copy()
    df_with_features = add_features(filtered_df, [selected_policy])

    trends_fig = plot_trends(df_with_features, selected_country, [selected_policy])

    prophet_df = prepare_data_for_prophet(df_with_features, selected_country)
    forecast = fit_and_predict_prophet(prophet_df)
    prophet_forecast_fig = plot_prophet_forecast(prophet_df, forecast, selected_country)

    t_test_results = perform_t_test(df_with_features, selected_policy)
    t_test_text = f"T-test results for {selected_policy} in {selected_country}: t-statistic = {t_test_results[selected_country]['t_stat']}, p-value = {t_test_results[selected_country]['p_value']}"

    return trends_fig, prophet_forecast_fig, t_test_text

if __name__ == "__main__":
    app.run_server(debug=True)