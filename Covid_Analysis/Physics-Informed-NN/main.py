import torch
import torch.optim as optim
import torch.nn as nn
from data_processing import main_preprocessing, prepare_tensors
from model import PINN
from training import train_model
from evaluation import evaluate_model, plot_results

if __name__ == "__main__":
    time, confirmed, recovered, deaths = main_preprocessing()
    t_train, S_train, I_train, R_train, t_val, S_val, I_val, R_val = prepare_tensors(time, confirmed, recovered, deaths)

    model = PINN()
    criterion = nn.MSELoss()  # We use MSE for smoother gradient during training
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, criterion, optimizer, t_train, S_train, I_train, R_train, t_val, S_val, I_val, R_val)

    # Evaluate the model
    test_loss = evaluate_model(model, criterion, t_val, S_val, I_val, R_val)

    # Plot the results
    plot_results(train_losses, val_losses)
