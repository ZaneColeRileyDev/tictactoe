import torch
import os
import numpy as np

from model import TicTacToeModel


def create_board() -> np.ndarray:
    board = np.zeros((3, 3), dtype=np.float32)
    return board


def load_model(path: str) -> TicTacToeModel:
    weights = torch.load(path, weights_only=True)
    model = TicTacToeModel()
    model.load_state_dict(weights)
    model.eval()
    return model


def ai_move(model, board: np.ndarray) -> np.ndarray:
    
    board_tensor = torch.tensor(board, dtype=torch.float32)

    with torch.no_grad():
        logits = model(board_tensor)
        logits = logits.flatten()
        mask = torch.tensor(board.flatten("C") != 0, dtype=torch.bool)

        logits = logits.masked_fill(mask, float("-inf"))

        idx = torch.argmax(logits).item()
        
    flat_board = board.flatten("C").copy()
    flat_board[idx] = 2
    return flat_board.reshape((3, 3))


def play_move(row_idx: int, col_idx: int, board: np.ndarray) -> np.ndarray:
    try:
        if board[row_idx, col_idx] != 0:
            return board
        else:
            board[row_idx, col_idx] = 1
            return board
    except IndexError:
        return board

def is_board_same(b1, b2) -> bool:
    if np.array_equal(b1, b2):
        return True
    else:
        return False


def check_win(p1, p2) -> list[bool, int]:
    if p1 == 2:
        return [True, 1]
    if p2 == 2:
        return [True, 2]
    else:
        p1 = 0
        p2 = 0
        return [False, 0]
def check_for_win(board: np.ndarray) -> list[bool, int]:
    lines = []
    lines.extend(board[r, :] for r in range(3))
    lines.extend(board[:, c] for c in range(3))
    lines.append(np.diag(board))
    lines.append(np.diag(np.fliplr(board)))

    for player in (1, 2):
        for line in lines:
            if np.all(line == player):
                return [True, player]
    return [False, 0]

def check_for_draw(board: np.ndarray) -> bool:
    if np.all(board != 0) and check_for_win(board=board)[0] is False:
        return True
    else:
        return False

def main():
    player = 1
    model_path = os.path.join("models", "TicTacToeModelV2.pth")
    board = create_board()
    model = load_model(model_path)
    
    while True:
        print("TicTacToe Console Game: Brought to you by ZGaming Studio's")
        print("Coded by ZaneColeRileyDev")
        print(board)
        if player > 2:
            player = 1

        board_past = board.copy()

        if player == 2:
            board = ai_move(model, board)
        if player == 1:
            row, col = str(input("Enter Row,Col: ")).split(",")
            board = play_move(row_idx=int(row), col_idx=int(col), board=board)

        same_board = is_board_same(b1=board, b2=board_past)
        if not same_board:
            player += 1
            is_win = check_for_win(board)
            winner = is_win[1] if is_win[0] else 0
            game_over = is_win[0] or check_for_draw(board)

            if game_over:
                if winner == 0:
                    os.system("cls")
                    print(f"Game is a Draw!!!")
                    print(board)
                    if input("Play Again? ").lower() == "yes":
                        main()
                    else:
                        exit()
                else:
                    os.system("cls")
                    print(f"Winner: P{winner}")
                    print(board)
                    if input("Play Again? ").lower() == "yes":
                        main()
                    else:
                        exit()
        os.system("cls")



if "__main__" == __name__:
    main()