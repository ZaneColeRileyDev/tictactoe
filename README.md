# TicTacToe AI

A console TicTacToe game where you play against a small neural network built with PyTorch. The model is trained to imitate the moves of winning players from a dataset of recorded games.

You play as **P1** and move first. The AI plays as **P2**.

## How It Works

**Model.** `TicTacToeModel` is a small fully connected network. It takes the 3x3 board, flattens it to 9 values, and outputs 9 logits, one per cell.

```
Input (9) -> Linear(64) -> ReLU -> Linear(18) -> ReLU -> Linear(9)
```

**Board encoding.**

| Value | Meaning |
|-------|---------|
| `0`   | Empty cell |
| `1`   | Player 1 (you) |
| `2`   | Player 2 (the AI) |

**Choosing a move.** The board goes through the model, any cell that is already occupied is masked out, and the AI plays the remaining cell with the highest score. It can never play an illegal move.

**Training.** The model is trained as a classifier with cross-entropy loss and the Adam optimizer. The dataset (`dataset.py`) keeps only rows where player 2 made the move **and** went on to win the game, so the model learns from winning moves. Each row is a board state (`cell_0` to `cell_8`) paired with the `action` taken (the cell index, 0-8).

## Project Structure

```
tictactoe/
├── app.py            # Console game: play against the trained model
├── model.py          # TicTacToeModel (PyTorch nn.Module)
├── dataset.py        # Parquet dataset loader + training log helper
├── training.py       # Training loop, checkpointing, validation
├── training_log.txt  # Log output from training runs
├── models/           # Final trained weights (TicTacToeModelV2.pth)
├── checkpoints/      # Per-epoch checkpoints (checkpoint_1.pth ... )
└── dataApp/
```

## Requirements

- Python 3.9+
- [PyTorch](https://pytorch.org/)
- NumPy
- Polars
- scikit-learn

Install the dependencies:

```bash
pip install torch numpy polars scikit-learn
```

## Play the Game

Make sure the trained weights exist at `models/TicTacToeModelV2.pth`, then run:

```bash
python app.py
```

Enter your move as `row,col`, using zero-based indices (0 to 2). For example, `1,1` plays the center square.

```
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]
Enter Row,Col: 1,1
```

After a win or draw, type `yes` at the prompt to play again.

> **Note:** The game clears the screen with `cls`, which works on Windows. On macOS or Linux, change `os.system("cls")` to `os.system("clear")` in `app.py`.

## Train Your Own Model

1. Place your dataset at `data/tictactoe_games.parquet`. It must contain these columns:
   - `cell_0` to `cell_8`: the board state before the move
   - `player`: which player made the move (1 or 2)
   - `winner`: which player won the game (1 or 2)
   - `action`: the cell index (0-8) that was played
2. Make sure the `models/` and `checkpoints/` folders exist.
3. Run:

   ```bash
   python training.py
   ```

Default training settings:

| Setting | Value |
|---------|-------|
| Epochs | 60 |
| Batch size | 32 |
| Train/validation split | 80/20 |
| Optimizer | Adam |
| Loss | Cross-entropy |
| Device | `cuda` |

Training writes loss values to `training_log.txt`, saves a checkpoint after every epoch to `checkpoints/`, and saves the final weights to `models/TicTacToeModelV2.pth`.

> **Note:** `training.py` uses `device = "cuda"`, so it needs an NVIDIA GPU by default. To train on CPU, change that line to `device = "cpu"`.

## Possible Improvements

- Train on both players' winning moves, or add self-play, for a stronger opponent
- Add a minimax opponent to benchmark the model's win/draw/loss rate
- Add a `requirements.txt` and a platform-independent screen clear
- Build a GUI or web front end

## Credits

Coded by [ZaneColeRileyDev](https://github.com/ZaneColeRileyDev), brought to you by ZGaming Studio's.

## License

No license has been specified yet. Consider adding one, such as [MIT](https://choosealicense.com/licenses/mit/).
