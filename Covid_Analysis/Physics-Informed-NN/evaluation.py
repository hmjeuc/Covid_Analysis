# evaluation.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def evaluate_model(model, criterion, t_test, S_test, I_test, R_test):
    """
    Evaluate the model performance.

    Args:
        model (nn.Module): The trained neural network model.
        criterion (nn.Module): Loss function.
        t_test (torch.Tensor): Test time tensor.
        S_test (torch.Tensor): Test susceptible cases tensor.
        I_test (torch.Tensor): Test infected cases tensor.
        R_test (torch.Tensor): Test recovered cases tensor.

    Returns:
        dict: Dictionary containing the test loss values.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(t_test).view(-1, 3)
        S_pred, I_pred, R_pred = outputs[:, 0], outputs[:, 1], outputs[:, 2]

        loss_S = criterion(S_pred, S_test.view(-1))
        loss_I = criterion(I_pred, I_test.view(-1))
        loss_R = criterion(R_pred, R_test.view(-1))

        beta = 0.3
        gamma = 0.1
        N = 1e6
        dS_dt = -beta * S_pred * I_pred / N
        dI_dt = beta * S_pred * I_pred / N - gamma * I_pred
        dR_dt = gamma * I_pred

        residual_loss = criterion(dS_dt + dI_dt + dR_dt, torch.zeros_like(dS_dt))

        total_loss = loss_S + loss_I + loss_R + residual_loss

    print(f'Test Loss: {total_loss.item()}')

    return {
        'S_loss': loss_S.item(),
        'I_loss': loss_I.item(),
        'R_loss': loss_R.item(),
        'residual_loss': residual_loss.item(),
        'total_loss': total_loss.item()
    }

def plot_results(train_losses, val_losses):
    """
    Plot the training and validation losses.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
    """
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
