import numpy as np
import random

board = np.zeros((4, 4)).astype(np.uint8)

board[0] = 1

def random_choice():
    rows, cols = np.where(board == 0)
    if len(rows) == 0:
        return None, None
    idx = random.randint(0, len(rows) - 1)
    return rows[idx].item(), cols[idx].item()

def squeeze(row):
    if np.sum(row != 0) == 0:
        return row
    # start = np.argmax(row != 0)
    new_row = np.zeros(4)
    set_idx = 0
    start = find_next_not_zero(row, 0)
    while start < len(row) and start is not None:
        next_idx = find_next_not_zero(row, start + 1)
        if next_idx is None:
            break
        if row[start] == row[next_idx]:
            new_row[set_idx] = row[start] * 2
            start = next_idx + 1
            set_idx += 1
        else:
            new_row[set_idx] = row[start]
            start = find_next_not_zero(row, next_idx)
            set_idx += 1
    if start < len(row):
        new_row[set_idx] = row[start]
    return new_row

def find_next_not_zero(row, idx):
    if idx >= len(row):
        return None
    if row[idx] != 0:
        return idx
    return find_next_not_zero(row, idx+1)

def generate_new():
    return 2 if random.random() < 0.9 else 4


reverse_squeeze = lambda row: squeeze(row[::-1])[::-1]

move_left = lambda board: np.apply_along_axis(squeeze, axis=1, arr=board)
move_right = lambda board: np.apply_along_axis(reverse_squeeze, axis=1, arr=board)
move_up = lambda board: np.apply_along_axis(squeeze, axis=0, arr=board)
move_down = lambda board: np.apply_along_axis(reverse_squeeze, axis=0, arr=board)

# row = np.array([2, 2, 2, 2])
# row2 = np.array([4, 2, 8, 2])
# row3 = np.array([0, 1, 1, 0])
# row4 = np.array([0, 0, 0, 1])



# squeeze()
# squeeze(row4)

# random_choice()

board = np.zeros((4, 4)).astype(np.uint8)
i, j = random_choice()
board[i][j] = generate_new()
while True:
    print(board)
    previous_board = board.copy()
    text = input("move?")
    if text == "u":
        board = move_up(board)
    if text == "l":
        board = move_left(board)
    if text == "r":
        board = move_right(board)
    if text == "d":
        board = move_down(board)
    if text == "q":
        break
    if np.all(previous_board == board):
        continue
    i, j = random_choice()
    if i is None:
        print("failed")
    else:
        board[i][j] = generate_new()