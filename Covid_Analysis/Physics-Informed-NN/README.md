# Physics-Informed Neural Networks (PINNs) for COVID-19 Modeling
# NOTE: THIS IS SIMPLY A PROJECT IDEA - NOT FULLY IMPLEMENTED!
## Overview 

This project utilizes Physics-Informed Neural Networks (PINNs) to model the spread of COVID-19. PINNs incorporate physical laws, expressed as differential equations, into the neural network training process. In this case, we use the SIR (Susceptible-Infectious-Recovered) model to describe the spread of the disease.

## Project Structure

- `data_processing.py`: Functions for loading, preprocessing, and normalizing the data.
- `model.py`: Definition of the PINN model architecture.
- `training.py`: Functions for preparing tensors and training the model.
- `evaluation.py`: Functions for evaluating the model and plotting results.
- `main.py`: Main script to run the entire pipeline.

## SIR Model and ODEs

The SIR model divides the population into three compartments:
- **S (Susceptible)**: Individuals who can contract the disease.
- **I (Infectious)**: Individuals who have contracted the disease and can spread it.
- **R (Recovered)**: Individuals who have recovered from the disease and are no longer infectious.

The model is governed by the following set of ordinary differential equations (ODEs):

**Note:** Rendering the equations in LaTeX syntax is not supported in markdown, but here are the equations:

- \(\frac{dS}{dt} = -\beta \frac{SI}{N}\)
- \(\frac{dI}{dt} = \beta \frac{SI}{N} - \gamma I\)
- \(\frac{dR}{dt} = \gamma I\)

Where:
- \(\beta\) is the transmission rate.
- \(\gamma\) is the recovery rate.
- \(N\) is the total population.

## Running the Project

1. Install the required packages:
   ```sh
   pip install -r requirements.txt
