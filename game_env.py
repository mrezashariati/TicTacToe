from dataclasses import dataclass, field
from typing import Literal, List, Callable, Any, Tuple, Dict
import numpy as np
from itertools import product
from tqdm import tqdm
import os
import pickle
from learning import MonteCarloEstimation, State, Action, Reward, ReplayBuffer
from utils import get_board_variations


@dataclass
class Player:
    mark: Literal["O", "X"]
    policy_type: Literal["manual", "RL"]
    # The policy is a function, which takes in a game state and outputs an action.
    # The action here is a position in the board on which the player wants to put his/her mark.
    _policy: Callable[[Any], Any] = field(init=False)

    def _init_policy(self):

        def random_policy(game_state):
            empty_states = np.where(game_state == 0)[0]
            return np.random.choice(empty_states, size=1)

        def manual_policy(game_state):
            empty_states = np.where(game_state == 0)[0]
            pos = int(input(f"choose the position. options: {empty_states}"))
            return pos

        if self.policy_type == "manual":
            return manual_policy

        elif self.policy_type == "random":
            return random_policy

        elif self.policy_type == "RL":
            return MonteCarloEstimation(raw_states=Board.load_all_valid_states())
        else:
            raise Exception(
                f"The defined policy doesn't exist. Pick from {['manual', 'random', 'RL']}"
            )

    def __post_init__(self):
        self._policy = self._init_policy()

    def play(self, game_state: np.ndarray):
        # Pick the position to put the mark
        return self._policy(game_state)  # pyright: ignore[reportOptionalCall]


@dataclass
class Board:
    players: List[Player]
    _positions: np.ndarray = field(
        default_factory=lambda: np.zeros(
            9,
        )
    )
    turn: int = 0  # always X starts the game
    finished: bool = False
    _winner: str | None = None
    _number_to_mark = {0: "-", 1: "O", 2: "X"}
    _mark_to_number: Dict[str, int] = field(init=False)

    def __post_init__(self):
        self.players.sort(key=lambda x: 0 if x.mark == "X" else 1)
        self._mark_to_number = {v: k for k, v in self._number_to_mark.items()}

    def get_reward(self, state: State, action: Action, mark: str) -> Reward:
        new_state_array = state.s.copy()
        new_state_array[action.a] = self._mark_to_number[mark]

        return Reward(1.0) if self._is_terminal(new_state_array) else Reward(0.0)

    @staticmethod
    def compute_all_valid_states():

        all_states = product([0, 1, 2], repeat=9)
        # TODO: an idea to improve the performance here is to save the states in a canonical form.
        # so when checking for the variations of the new state, first make it canonical and then
        # check for it in previous states.
        states = []
        for s in tqdm(list(all_states)):
            xcount = s.count(2)
            ocount = s.count(1)
            # TODO: both players cannot be winners
            if xcount - ocount in [0, 1]:
                # state is valid, then check for the rotations
                new_state = np.array(s).reshape(3, 3)
                if len(states) == 0:
                    states.append(new_state)
                    continue
                variations = get_board_variations(new_state)
                if not np.any(
                    [np.array_equal(v, arr) for v in variations for arr in states]
                ):
                    states.append(new_state)

        return states

    @staticmethod
    def save_all_valid_states(states):
        with open("all_valid_states.pkl", "wb") as f:
            pickle.dump(states, f)

    @staticmethod
    def load_all_valid_states():
        if os.path.exists("all_valid_states.pkl"):
            print("Saved states found. Loading...")
            with open("all_valid_states.pkl", "rb") as f:
                states = pickle.load(f)
            return states
        else:
            print("No saved states. Computing...")
            states = Board.compute_all_valid_states()
            Board.save_all_valid_states(states)
            print("Saved all valid states")
            return states

    @staticmethod
    def _is_terminal(positions: np.ndarray) -> Tuple[bool, str | None]:
        # TODO: simplify this
        sequences = [
            list(range(0, 3, 1)),
            list(range(3, 6, 1)),
            list(range(6, 9, 1)),
            list(range(0, 7, 3)),
            list(range(1, 8, 3)),
            list(range(2, 9, 3)),
            list(range(0, 9, 4)),
            list(range(2, 7, 2)),
        ]
        finished = False
        winner = None
        for seq in sequences:
            if all(positions[seq] == 1):
                finished = True
                winner = "O"
                return finished, winner
            elif all(positions[seq] == 2):
                finished = True
                winner = "X"
                return finished, winner

        finished = not np.any(positions == 0)
        return finished, winner

    def get_game_state(self):
        return self._positions.copy()

    def pretty_print_board(self):
        display_board = self._positions.reshape(3, 3).copy().astype(int).astype(str)
        for k in self._number_to_mark:
            display_board[np.where(display_board == str(k))] = self._number_to_mark[k]
        for r in display_board:
            print("|".join(list(r)))

    def put_mark(self, pos):
        assert (
            not self._positions[pos] != 0
        ), "position already full, cannot put mark there bro"
        if not self.finished:
            self._positions[pos] = 1 if self.players[self.turn].mark == "O" else 2
            self.turn = int(not bool(self.turn))
            print(f"player {self.players[self.turn].mark}'s turn:")
            self.pretty_print_board()
            self.finished, self._winner = self._is_terminal(self._positions)
            if self.finished:
                print(
                    "Game finished. The winner:",
                    self._winner if self._winner else "Neither. Its an equal.",
                )
                return
        else:
            raise Exception("game already finished. cannot make any more moves")
