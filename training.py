import torch
import os
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from dataset import TicTacToeDataset, write_training_log
from model import TicTacToeModel
from sklearn.model_selection import train_test_split

if "__main__" == __name__:

    device = "cuda"

    data_path = os.path.join("data", 'tictactoe_games.parquet')

    ds = TicTacToeDataset(data_path)

    ds_size = len(ds)
    indices = list(range(ds_size))
    batch_size = 32
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=12)

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = TicTacToeModel().to(device=device)
    n_epochs = 60
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters())

    for t in range(n_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            write_training_log(f"Epoch {t+1}\n-------------------------------")
            model.train()
            size = len(train_ds)
            for batch, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                y_pred = model(inputs)
                # print(y_pred.shape, labels.shape, labels.dtype)
                loss = criterion(y_pred, labels.squeeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * 32 + len(inputs)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                    write_training_log(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            model.eval()
            size = len(val_loader.dataset)
            num_batches = len(val_loader)
            test_loss = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    pred = model(inputs)
                    test_loss += criterion(pred, labels.squeeze(1)).item()
                test_loss /= num_batches
                print(f"Validation loss: {test_loss:>8f}\n")
                write_training_log(f"Validation loss: {test_loss:>8f}\n")
            checkpoint = {
                "epoch": t + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss
                    }
            torch.save(checkpoint, f"checkpoints/checkpoint_{t + 1}.pth")
    torch.save(model.state_dict(), "models/TicTacToeModelV2.pth")
