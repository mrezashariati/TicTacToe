# here lies the learning algorithm inlcuding the policy, action taking and value updating

from dataclasses import dataclass, field, InitVar
from typing import List, Tuple, Dict, Any
import numpy as np
from numpy.typing import NDArray
import random

np.random.seed(42)
random.seed(42)


# State and Action space of TicTacToe
@dataclass
class State:
    s: NDArray[np.int16]

    def __hash__(self) -> int:
        return hash("".join(map(str, self.s.tolist())))

    def __eq__(self, other) -> bool:
        if isinstance(other, State):
            return all(self.s == other.s)

        return False


@dataclass
class Action:
    a: int

    def __hash__(self) -> int:
        return hash(str(self.a))

    def __eq__(self, other) -> bool:
        if isinstance(other, Action):
            return self.a == other.a

        return False


@dataclass
class Reward:
    r: float


@dataclass
class ReplayBuffer:
    buffer: List[List[Tuple[State, Action, Reward]]]


@dataclass
class MonteCarloEstimation:
    raw_states: InitVar[List[NDArray[np.float16]]]
    # All valid states of the XO
    states: List[State] = field(init=False)

    # an action is putting the mark on a position of the table. For XO, we have 9 positions, hence 9 actions.
    # not all actions are valid for a particular state
    actions: List[Action] = field(default_factory=lambda: [Action(i) for i in range(9)])
    Q_values: Dict[Tuple[State, Action], float] = field(init=False)
    discount_factor = 0.9

    def __post_init__(self, raw_states):

        self.states = [State(s.reshape(-1)) for s in raw_states]

        # Init the Q values. The row is state index and column is action index
        self.Q_values = {
            (s, a): np.random.random() for s in self.states for a in self.actions
        }

        print(
            f"number of actions: {len(self.actions)}, number of states: {len(self.states)}, number (state,action)s: {len(self.Q_values)}"
        )

    def find_state(self, s: State) -> State | None:
        pass

    # This computes discounted sum of future rewards.
    def discounted_return(self, rewards):
        return 0

    def estimate_qvalues(self, buffer: ReplayBuffer) -> None:
        # iterate over the replay buffer and update Q values
        pass

    def __call__(self, state_array: NDArray[Any]) -> Action:
        # return the best action based on the Q-values stored
        state = State(state_array.astype(int))  # type: ignore
        # TODO: the state is not in canonical form and only an equivalant state maybe in the states list. Make sure you find it
        best_action_value = -float("inf")
        best_action = None
        for a in self.actions:
            if self.Q_values[(state, a)] > best_action_value:
                best_action = a
        return best_action.a  # type: ignore


# This class does the self play and sample generation. It can be used for any game with two players
@dataclass
class SelfPlay:
    n_episodes: int
