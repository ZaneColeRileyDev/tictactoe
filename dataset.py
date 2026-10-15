import polars as pl
import os
import torch

from torch.utils.data import Dataset


def write_training_log(text) -> None:
    with open("training_log.txt", "a") as f:
        f.write(text)
        f.write("\n")

class TicTacToeDataset(Dataset):
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError("Data File Doesn't Exist!!")

        if not path.endswith(".parquet"):
            raise ValueError("File Type is incorrect, it must be parquet!")

        df = pl.read_parquet(path)

        filtered_df = df.filter((pl.col("player") == 2) & (pl.col("winner") == 2))

        self.X = filtered_df.select(["cell_0", "cell_1", "cell_2", "cell_3", "cell_4",
                                 "cell_5", "cell_6", "cell_7", "cell_8"])
        self.X = torch.tensor(self.X.to_numpy().reshape(len(self.X), 3, 3), dtype=torch.float32)

        self.y = filtered_df.select("action")

        self.y = torch.tensor(self.y.to_numpy(), dtype=torch.long)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        idx = int(idx)
        return self.X[idx], self.y[idx]


     