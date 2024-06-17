# COVID-19 Dashboard

## Overview

This project aims to inspect the possible impact of certain COVID-19 policies on the spread of the virus using various data sources. The project merges data from the Oxford COVID-19 Government Response Tracker (OxCGRT), COVID-19 case data, and population data to provide insights into how different policies affect the spread of COVID-19.

## Data Sources

- [OxCGRT](https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_nat_latest.csv)
- [Confirmed Cases](https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv)
- [Deaths](https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv)
- [Recovered](https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv)
- [Population](https://raw.githubusercontent.com/datasets/population/master/data/population.csv)

## Features

- **Daily New Cases**: Difference in confirmed cases from the previous day.
- **Smoothed Daily New Cases**: 7-day rolling average of daily new cases.
- **Daily Growth Rate**: Percentage change in smoothed daily new cases.
- **R0**: Reproduction number calculated from the daily growth rate.
- **Policy Implementation**: Indicator of whether a policy is active.

## Analysis Methods

- **Prophet Model**: Used to forecast the reproduction number (R0) over time.
- **T-test**: Used to compare daily growth rates before and after policy implementation.

## Running the Project

1. Install the required packages:
   ```sh
   pip install -r requirements.txt
## Future Work

### Interrupted Time Series Analysis (ITSA)

ITSA is a statistical technique used to evaluate the impact of an intervention by comparing the trends before and after the intervention. It helps in identifying whether the intervention (e.g., policy implementation) caused a significant change in the observed outcome (e.g., infection rate). Future work will involve implementing ITSA to provide a more robust analysis of the impact of COVID-19 policies.

### Physics-Informed Neural Networks (PINNs)

PINNs incorporate physical laws, expressed as differential equations, into the neural network training process. For modeling the spread of COVID-19, we can use the SIR (Susceptible-Infectious-Recovered) model, which is a type of compartmental model used to describe the spread of diseases.

#### SIR Model and ODEs

The SIR model divides the population into three compartments:
- **S (Susceptible):** Individuals who can contract the disease.
- **I (Infectious):** Individuals who have contracted the disease and can spread it.
- **R (Recovered):** Individuals who have recovered from the disease and are no longer infectious.

The model is governed by the following set of ordinary differential equations (ODEs):


#### Note: LaTeX rendering is not supported and still presents the equations in LaTeX format for reference. Sorry :) ####

\frac{dS}{dt} &= -\beta \frac{SI}{N} \\
\frac{dI}{dt} &= \beta \frac{SI}{N} - \gamma I \\
\frac{dR}{dt} &= \gamma I

Where:
- \( \beta \) is the transmission rate.
- \( \gamma \) is the recovery rate.
- \( N \) is the total population.

PINNs will be trained to solve these ODEs, which ensures that the learned model adheres to the known physical laws of disease transmission. This can potentially improve the accuracy and reliability of the predictions.
