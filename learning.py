# here lies the learning algorithm inlcuding the policy, action taking and value updating

from dataclasses import dataclass, field, InitVar
from typing import List, Tuple, Dict, Any
import numpy as np
from numpy.typing import NDArray
import random
from collections import defaultdict
from utils import get_board_variations
from entities import State, Action

# np.random.seed(42)
# random.seed(42)


@dataclass
class Reward:
    r: float


@dataclass
class ReplayBuffer:
    buffer: List[List[Tuple[State, Action, Reward]]]

    def __len__(self):
        return len(self.buffer)


@dataclass
class MonteCarloEstimation:
    raw_states: InitVar[List[NDArray[np.float16]]]
    # All valid states of the XO
    states: Dict[Tuple[np.intp, np.intp], List[State]] = field(init=False)

    # an action is putting the mark on a position of the table. For XO, we have 9 positions, hence 9 actions.
    # not all actions are valid for a particular state
    actions: List[Action]
    Q_values: Dict[Tuple[State, Action], float] = field(init=False)
    discount_factor = 0.1
    learning_rate = 0.8
    epsilon = 0.60
    eval_mode: bool = False

    def __post_init__(self, raw_states: List[NDArray[Any]]):
        self.states = defaultdict(list)
        for s in raw_states:
            xcount = np.count_nonzero(s == 2)
            ocount = np.count_nonzero(s == 1)
            self.states[(xcount, ocount)].append(State(s))

        # Init the Q values. The row is state index and column is action index
        # TODO: This nested loop seems nasty brooo
        self.Q_values = {
            (s, a): 0.0 for v in self.states.values() for s in v for a in self.actions
        }

    def eval(self):
        # eval mode disables epsilon-greedy. Only greedy behaviour.
        self.eval_mode = True

    def train(self):
        # train mode enables epsilon-greedy behaviour.
        self.eval_mode = False

    def find_state(self, s: State) -> State | None:
        xcount = np.count_nonzero(s.s == 2)
        ocount = np.count_nonzero(s.s == 1)
        candidate_states = self.states[(xcount, ocount)]
        board_variations = get_board_variations(s.s, include_self=True)

        # search for board variations in candidate states:
        for b in board_variations:
            for c in candidate_states:
                if State(b) == c:
                    return c

        return None

    # This computes discounted sum of future rewards.
    def discounted_return(self, rewards):
        return sum([self.discount_factor**i * r for i, r in enumerate(rewards)])

    def estimate_qvalues(self, buffer: ReplayBuffer, player_mark) -> None:
        # iterate over the replay buffer and update Q values
        # only take into account actions taken by this player (X or O)

        for e in buffer.buffer:
            for i, (s, a, _) in enumerate(e):
                if a.mark != player_mark:
                    continue

                future_rewards = [
                    rt.r for (_, at, rt) in e[i:] if at.mark == player_mark
                ]
                q_estimate = self.discounted_return(future_rewards)

                canonical_state = self.find_state(s)
                if not canonical_state:
                    raise Exception("didn't find the state :(( Why?")

                q_oldval = self.Q_values[(canonical_state, a)]

                self.Q_values[(canonical_state, a)] = q_oldval + self.learning_rate * (
                    q_estimate - q_oldval
                )

    def get_state_values(self, state: State) -> List[Tuple[float, Action]]:
        canonical_state = self.find_state(state)
        assert canonical_state, "couldn't find the canonical state. Thats not good bro."
        values = [self.Q_values[(canonical_state, a)] for a in self.actions]
        return list(zip(values, self.actions))

    def __call__(self, state: State) -> Action:
        """returns the best action based on the Q-values stored with 1-epsilon probability"""
        possible_actions = [a for a in self.actions if state.s.reshape(-1)[a.pos] == 0]

        # epsilon-greedy. Take random action with 1-epsilon prob if in train mode
        if not self.eval_mode and np.random.rand() > self.epsilon:
            return possible_actions[np.random.choice(np.arange(len(possible_actions)))]

        state = State(state.s.reshape(3, 3).astype(int))  # type: ignore

        # find the canonical form of the state
        canonical_state = self.find_state(state)

        if canonical_state is None:
            raise Exception("couldn't find the state, it shouldn't be missing :/")

        best_action_value = -float("inf")
        best_action = None
        for a in possible_actions:
            if self.Q_values[(canonical_state, a)] > best_action_value:
                best_action = a
        return best_action  # type: ignore
