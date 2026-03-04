from dataclasses import dataclass, field
from typing import Literal, List, Callable, Any, Tuple, Dict
import numpy as np
from itertools import product
from tqdm import tqdm
import os
import pickle
from learning import MonteCarloEstimation, Reward, ReplayBuffer
from utils import get_board_variations
from entities import State, Action
import logging
import random


@dataclass
class Player:
    mark: Literal["O", "X"]
    policy_type: Literal["manual", "rl", "random", "gfi"]
    possible_actions: List[Action] = field(init=False)

    # The policy is a function, which takes in a game state and outputs an action.
    # The action here is a position in the board on which the player wants to put his/her mark.
    _policy: Callable[[Any], Any] = field(init=False)

    def _init_policy(self):

        def random_policy(game_state: State):
            empty_states = np.where(game_state.s == 0)[0]
            pos = int(np.random.choice(empty_states, size=1)[0])
            return Action(pos=pos, mark=self.mark)

        def manual_policy(game_state: State):
            empty_states = np.where(game_state.s == 0)[0]
            pos = int(input(f"choose the position. options: {empty_states} "))
            return Action(pos=pos, mark=self.mark)

        def go_for_it_policy(game_state: State):
            # in this policy the player just goes for it.
            # The first free line it sees, it starts putting marks on
            # This policy wins in 72% of times against random policy, when played as "O"

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

            for s in sequences:
                # find the first applicable sequence
                if all(
                    [
                        game_state.s[i] in [0, Board._mark_to_number[self.mark]]
                        for i in s
                    ]
                ):
                    # find the first free spot
                    for pos in s:
                        if game_state.s[pos] == 0.0:
                            return Action(pos=pos, mark=self.mark)

            # couldn't find any winning sequence, return random
            empty_states = np.where(game_state.s == 0)[0]
            pos = int(np.random.choice(empty_states, size=1)[0])
            return Action(pos=pos, mark=self.mark)

        match self.policy_type:
            case "manual":
                return manual_policy
            case "random":
                return random_policy
            case "rl":
                return MonteCarloEstimation(
                    raw_states=Board.load_all_valid_states(),
                    actions=self.possible_actions,
                )
            case "gfi":
                return go_for_it_policy
            case _:
                raise Exception(
                    f"The defined policy doesn't exist. Pick from {['manual', 'random', 'rl']}"
                )

    def __post_init__(self):
        self.possible_actions = [Action(i, self.mark) for i in range(9)]
        self._policy = self._init_policy()

    def play(self, game_state: State):
        # Pick the position to put the mark
        return self._policy(game_state)  # pyright: ignore[reportOptionalCall]

    def learn(self, data: ReplayBuffer) -> None:
        if self.policy_type in ["manual", "random"]:
            raise Exception(f"cannot learn with policy {self.policy_type}")

        self._policy.estimate_qvalues(data, self.mark)
        return

    def eval(self):
        # only the players with RL policy can enable eval mode
        if self.policy_type == "rl":
            self._policy.eval()

    def train(self):
        # only the players with RL policy can enable train mode
        if self.policy_type == "rl":
            self._policy.train()


@dataclass
class Board:
    # TODO: there is no turn here, so it is better to have a restriction on possible actions. Priority: low
    _positions: np.ndarray = field(
        default_factory=lambda: np.zeros(
            9,
        )
    )
    finished: bool = False
    _winner: str | None = None
    _number_to_mark = {0: "-", 1: "O", 2: "X"}
    _mark_to_number = {"-": 0, "O": 1, "X": 2}

    def __post_init__(self):
        pass

    def get_reward(self, state: State, action: Action) -> Dict[str, Reward]:
        # Here, the reward is decoupled from the action of each player.
        # Meaning, the reward can come in any time/step of the game. For this reason.
        # For this reason, one player can do nothing and still get a reward.
        # For example, when the opponent does a move and wins the game, the player gets a -1 reward.

        new_state_array = state.copy()
        new_state_array.s[action.pos] = self._mark_to_number[action.mark]

        f = self.is_terminal(new_state_array)
        w = self.get_winner(new_state_array)

        rewards = {"O": Reward(0.0), "X": Reward(0.0)}
        if f and w == "X":
            rewards["X"], rewards["O"] = Reward(1.0), Reward(-1.0)
        elif f and w == "O":
            rewards["O"], rewards["X"] = Reward(1.0), Reward(-1.0)

        return rewards

    @staticmethod
    def compute_all_valid_states():

        all_states = product([0, 1, 2], repeat=9)
        # TODO: an idea to improve the performance here is to save the states in a canonical form.
        # so when checking for the variations of the new state, first make it canonical and then
        # check for it in previous states. Priority: Medium
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
            logging.info("Saved states found. Loading...")
            with open("all_valid_states.pkl", "rb") as f:
                states = pickle.load(f)
            return states
        else:
            logging.info("No saved states. Computing...")
            states = Board.compute_all_valid_states()
            Board.save_all_valid_states(states)
            logging.info("Saved all valid states")
            return states

    @staticmethod
    def get_winner(state: State) -> str | None:
        # this function is implemented very much the is_terminal func.
        # it only returns the winner if exists.

        flat_state = State.flat(state)

        # TODO: simplify this. Priority: low
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
        for seq in sequences:
            if all(flat_state.s[seq] == 1):
                return "O"
            elif all(flat_state.s[seq] == 2):
                return "X"

        return

    @staticmethod
    def is_terminal(state: State) -> bool:

        flat_state = State.flat(state)

        # TODO: simplify this. Priority: Low
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
        for seq in sequences:
            if all(flat_state.s[seq] == 1):
                return True
            elif all(flat_state.s[seq] == 2):
                return True

        finished = not np.any(flat_state.s == 0)
        return finished

    def get_game_state(self) -> State:
        return State(s=self._positions.copy())

    def pretty_print_board(self):
        display_board = self._positions.reshape(3, 3).copy().astype(int).astype(str)
        for k in self._number_to_mark:
            display_board[np.where(display_board == str(k))] = self._number_to_mark[k]
        for r in display_board:
            logging.info("|".join(list(r)))

    def step(self, action: Action) -> Dict[str, Reward]:
        # upack the action
        pos, mark = action.pos, action.mark
        assert not self._positions[pos] != 0, f"Action not possbile {action}"
        if not self.finished:
            # get the reward of the action in current state
            rewards = self.get_reward(self.get_game_state(), action)
            # carry out the action
            self._positions[pos] = 1 if mark == "O" else 2
            # self.pretty_print_board()
            self.finished = self.is_terminal(self.get_game_state())
            self._winner = self.get_winner(self.get_game_state())

            return rewards
        else:
            raise Exception("game already finished. cannot make any more moves")


@dataclass
class GameRunner:
    num_episodes: int
    env: Any  # TODO: what is the type here?
    players: List[Player]
    _winners: List[str] = field(default_factory=list)
    _episodes: ReplayBuffer = field(init=False)

    def __post_init__(self):
        # this is not generalizable. TicTacToe always starts with X
        self.players.sort(key=lambda x: 0 if x.mark == "X" else 1)
        # put all players in train mode
        for player in self.players:
            player.train()

    def run(self) -> None:
        game_episodes = []
        for _ in range(self.num_episodes):
            # initiate a new game
            game = self.env()
            turn = self.players[0]
            new_episode = []
            while not game.finished:
                # get current game state
                state = game.get_game_state()

                # get the player's action for current state
                action = turn.play(state)

                # get the reward for current (state, action) and carry out the action
                rewards = game.step(action)

                # change the turn
                turn = self.players[0] if turn != self.players[0] else self.players[1]

                # log
                new_episode.append((state, action, rewards))

            winner = game._winner if game._winner else "D"
            self._winners.append(winner)
            game_episodes.append(new_episode)

        self._episodes = ReplayBuffer(game_episodes)

        return

    def get_generated_episodes(self):
        return self._episodes

    def get_winners(self):
        return self._winners
