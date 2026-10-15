import numpy as np
import polars as pl
import os

from tqdm import tqdm

def create_board() -> np.ndarray:
    board = np.zeros((3, 3), dtype=np.int16)
    return board






def create_game_data(board_data: list, game_num: int) -> None:
    df = pl.DataFrame(board_data)
    df.write_csv(f"data/tictactoeGame_{game_num}.csv")



def play_move(player: int, row_idx: int, col_idx: int, board: np.ndarray) -> np.ndarray:
    if player == 1:
        if board[row_idx, col_idx] != 0:
            return board
        else:
            board[row_idx, col_idx] = 1
            return board
    else:
        if board[row_idx, col_idx] != 0:
            return board
        else:
            board[row_idx, col_idx] = 2
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


def auto_play(board):
    empties = np.argwhere(board == 0)
    row, col = empties[np.random.randint(len(empties))]
    return int(row), int(col)

def extract_board_data(board: np.ndarray, board_data: list, turn: int, player: int, r: int, col: int) -> list:
    row = {"turn": turn, "player": player, "action": r * 3 + col}
    for i, value in enumerate(board.flatten().tolist()):
        row[f"cell_{i}"] = value
    board_data.append(row)
    return board_data

    

games = 10000

def main():
    board = create_board()
    player = 1
    board_data = []
    games_played = 0
    moves_num = 0
    all_games = []
    with tqdm(total=games, desc="Generating Games") as pbar:
        while games_played < games:
            if player > 2:
                player = 1
            row, col = auto_play(board=board)
            board_past = board.copy()
            board = play_move(player=player, row_idx=row, col_idx=col, board=board)
            same_board = is_board_same(b1=board, b2=board_past)
            if not same_board:
                board_data = extract_board_data(board=board_past, board_data=board_data, turn=moves_num, player=player, r=row, col=col)
                moves_num += 1
                player += 1
                is_win = check_for_win(board)
                winner = is_win[1] if is_win[0] else 0
                game_over = is_win[0] or check_for_draw(board)

                if game_over:
                    all_games.extend(board_data)
                    for r in board_data:
                        r["game"] = games_played
                        r["winner"] = winner
                    board = create_board()
                    board_data = []
                    games_played += 1
                    player = 1
                    moves_num = 0
                    pbar.update(1)
            else:
                continue

    pl.DataFrame(all_games).write_parquet("data/tictactoe_games.parquet")
    print("Data Generated")
    
    


if "__main__" == __name__:
    main()