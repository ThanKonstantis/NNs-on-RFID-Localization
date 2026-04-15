import torch
import torch.nn as nn

from src.training.callbacks import EarlyStopping


def train_step(model, data_loader, loss_fn, optimizer, device="cpu"):
    model.to(device)
    model.train()
    train_loss = 0
    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    return train_loss


def test_step(model, data_loader, loss_fn, device="cpu"):
    model.to(device)
    model.eval()
    test_loss = 0
    with torch.inference_mode():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            test_loss += loss_fn(model(X_batch), y_batch).item()
    return test_loss


def train_test_model(epoch, model, train_loader, test_loader, loss_fn, optimizer,
                     scheduler, early_stopper=None, device="cpu", test=True, verbose=True):
    torch.manual_seed(42)
    train_loss_arr = []
    test_loss_arr = []

    if test:
        for epochs in range(epoch):
            epoch_loss = train_step(model, train_loader, loss_fn, optimizer, device)
            train_loss_arr.append(epoch_loss / len(train_loader))
            scheduler.step(epoch_loss)
            lr = optimizer.param_groups[0]["lr"]

            test_loss = test_step(model, test_loader, loss_fn, device)
            test_loss_arr.append(test_loss / len(test_loader))

            if verbose:
                if epochs % 10 == 0 or epochs == 0 or epochs + 1 == epoch:
                    print(f"Epoch {epochs} | train={train_loss_arr[-1]:.6f} | test={test_loss_arr[-1]:.6f} | lr={lr}")

            if early_stopper is not None:
                early_stopper(test_loss_arr[-1], model)
                if early_stopper.early_stop:
                    if verbose:
                        print(f"Early stopping at epoch {epochs}")
                    break

        if early_stopper is not None and early_stopper.path is not None:
            model.load_state_dict(torch.load(early_stopper.path, weights_only=True))
            if verbose:
                print(f"Loaded best model from {early_stopper.path}")

        best_loss = early_stopper.get_best_loss() if early_stopper else test_loss_arr[-1]
        return train_loss_arr, test_loss_arr, best_loss, epochs
    else:
        for epochs in range(epoch):
            epoch_loss = train_step(model, train_loader, loss_fn, optimizer, device)
            train_loss_arr.append(epoch_loss / len(train_loader))
            scheduler.step(epoch_loss)
            lr = optimizer.param_groups[0]["lr"]
            if verbose and (epochs % 10 == 0 or epochs == 0 or epochs + 1 == epoch):
                print(f"Epoch {epochs} | train={train_loss_arr[-1]:.6f} | lr={lr}")
        return train_loss_arr
