import numpy as np


def get_board_variations(board, include_self=False):
    if board.shape != (3, 3):
        board = board.reshape(3, 3)

    variations = {
        "hf": np.flip(board, axis=0),  # horizontal flip
        "vf": np.flip(board, axis=1),  # vertical flip
        "rot1": np.rot90(board, 1),  # rotation
        "rot2": np.rot90(board, 2),  # rotations
        "rot3": np.rot90(board, 3),  # rotations
        "dm1": board.T,  # diagonal mirror
        "dm2": np.rot90(board, k=2).T,  # diagonal mirror
    }
    if include_self:
        variations["id"] = board

    return variations


def get_position_inverse_transformed(transformation: str, position: int) -> int:
    # This function returns the position in the board, after applying the inverse transformation
    # This func is used to go from canonical coordinate to normal board coordinate
    board = np.zeros((3, 3)).reshape(-1)
    board[position] = 1
    board = board.reshape(3, 3)

    match transformation:
        case "hf":
            # the inverse of horizontal flip is itself.
            board = np.flip(board, axis=0)
        case "vf":
            # the inverse of vertical flip is itself
            board = np.flip(board, axis=1)
        case "rot1":
            board = np.rot90(board, -1)
        case "rot2":
            board = np.rot90(board, -2)
        case "rot3":
            board = np.rot90(board, -3)
        case "dm1":
            board = board.T
        case "dm2":
            board = np.rot90(board.T, k=-2)
        case "id":
            board = board
        case _:
            raise Exception("invalid transformation!")

    board = board.reshape(-1)
    position_transformed = np.where(board == 1)[0][0]

    return position_transformed


def get_position_transformed(transformation: str, position: int) -> int:
    # This function returns the position in the board, after applying the transformation
    # This func is used to go from normal board coordinate to canonical coordinate
    board = np.zeros((3, 3)).reshape(-1)
    board[position] = 1
    board = board.reshape(3, 3)

    match transformation:
        case "hf":
            # horizontal flip
            board = np.flip(board, axis=0)
        case "vf":
            # vertical flip
            board = np.flip(board, axis=1)
        case "rot1":
            board = np.rot90(board, 1)
        case "rot2":
            board = np.rot90(board, 2)
        case "rot3":
            board = np.rot90(board, 3)
        case "dm1":
            board = board.T
        case "dm2":
            board = np.rot90(board, k=2).T
        case "id":
            board = board
        case _:
            raise Exception("invalid transformation!")

    board = board.reshape(-1)
    position_transformed = np.where(board == 1)[0][0]

    return position_transformed
