import numpy as np


def get_board_variations(board, include_self=False):
    if board.shape != (3, 3):
        board = board.reshape(3, 3)
        
    variations = [
        np.flip(board, axis=0),  # horizontal flip
        np.flip(board, axis=1),  # vertical flip
        *[np.rot90(board, i) for i in range(1, 4)],  # rotations
        board.T,  # diagonal mirror
        np.rot90(board, k=2).T,  # diagonal mirror
    ]
    if include_self:
        variations.append(board)

    return variations
