import torch
import torch.optim as optim
import matplotlib.pyplot as plt

def prepare_tensors(time, confirmed, recovered, deaths):
    """
    Prepare tensors for training and validation.

    Args:
        time (np.array): Array of time steps.
        confirmed (np.array): Array of confirmed cases.
        recovered (np.array): Array of recovered cases.
        deaths (np.array): Array of death cases.

    Returns:
        tuple: Tensors for training and validation datasets.
    """
    t_train = torch.tensor(time, dtype=torch.float32).view(-1, 1)
    S_train = torch.tensor(confirmed, dtype=torch.float32).view(-1, 1)
    I_train = torch.tensor(recovered, dtype=torch.float32).view(-1, 1)
    R_train = torch.tensor(deaths, dtype=torch.float32).view(-1, 1)

    train_size = int(0.8 * len(t_train))
    val_size = len(t_train) - train_size

    t_train, t_val = torch.split(t_train, [train_size, val_size])
    S_train, S_val = torch.split(S_train, [train_size, val_size])
    I_train, I_val = torch.split(I_train, [train_size, val_size])
    R_train, R_val = torch.split(R_train, [train_size, val_size])

    return t_train, S_train, I_train, R_train, t_val, S_val, I_val, R_val

def train_model(model, criterion, optimizer, t_train, S_train, I_train, R_train, t_val, S_val, I_val, R_val, epochs=40000):
    """
    Train the model with early stopping.

    Args:
        model (nn.Module): The neural network model.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        t_train (torch.Tensor): Training time tensor.
        S_train (torch.Tensor): Training susceptible cases tensor.
        I_train (torch.Tensor): Training infected cases tensor.
        R_train (torch.Tensor): Training recovered cases tensor.
        t_val (torch.Tensor): Validation time tensor.
        S_val (torch.Tensor): Validation susceptible cases tensor.
        I_val (torch.Tensor): Validation infected cases tensor.
        R_val (torch.Tensor): Validation recovered cases tensor.
        epochs (int): Number of epochs for training.
    """
    early_stopping_patience = 10
    min_delta = 0.0001
    patience_counter = 0
    best_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(t_train).view(-1, 3)
        S_pred, I_pred, R_pred = outputs[:, 0], outputs[:, 1], outputs[:, 2]

        loss_S = criterion(S_pred, S_train.view(-1))
        loss_I = criterion(I_pred, I_train.view(-1))
        loss_R = criterion(R_pred, R_train.view(-1))

        beta = 0.3
        gamma = 0.1
        N = 1e6
        dS_dt = -beta * S_pred * I_pred / N
        dI_dt = beta * S_pred * I_pred / N - gamma * I_pred
        dR_dt = gamma * I_pred

        residual_loss = criterion(dS_dt + dI_dt + dR_dt, torch.zeros_like(dS_dt))

        loss = loss_S + loss_I + loss_R + residual_loss
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_outputs = model(t_val).view(-1, 3)
            S_val_pred, I_val_pred, R_val_pred = val_outputs[:, 0], val_outputs[:, 1], val_outputs[:, 2]

            val_loss_S = criterion(S_val_pred, S_val.view(-1))
            val_loss_I = criterion(I_val_pred, I_val.view(-1))
            val_loss_R = criterion(R_val_pred, R_val.view(-1))
            val_loss = val_loss_S + val_loss_I + val_loss_R

            val_losses.append(val_loss.item())

        if val_loss.item() < best_loss - min_delta:
            best_loss = val_loss.item()
            patience_counter = 0
            torch.save(model.state_dict(), '/content/pinn_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')

    torch.save(model.state_dict(), '/content/pinn_model_final.pth')
    print("Final model saved to /content/pinn_model_final.pth")

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
