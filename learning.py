# here lies the learning algorithm inlcuding the policy, action taking and value updating

from dataclasses import dataclass, field, InitVar
from typing import List, Tuple, Dict, Any
import numpy as np
from numpy.typing import NDArray
import random
from collections import defaultdict
from utils import (
    get_board_variations,
    get_position_inverse_transformed,
    get_position_transformed,
)
from entities import State, Action

# np.random.seed(42)
# random.seed(42)


@dataclass
class Reward:
    r: float


@dataclass
class ReplayBuffer:
    buffer: List[List[Tuple[State, Action, Dict[str, Reward]]]]

    def __len__(self):
        return len(self.buffer)


@dataclass
class MonteCarloEstimation:
    raw_states: InitVar[List[NDArray[np.float16]]]
    # All valid states of the XO. The state's board's shape is (3, 3)
    states: Dict[Tuple[np.intp, np.intp], List[State]] = field(init=False)

    # an action is putting the mark on a position of the table. For XO, we have 9 positions, hence 9 actions.
    # not all actions are valid for a particular state
    actions: List[Action]
    Q_values: Dict[Tuple[State, Action], float] = field(init=False)
    discount_factor = 0.7
    learning_rate = 0.4
    epsilon = 0.6
    eval_mode: bool = False

    def __post_init__(self, raw_states: List[NDArray[Any]]):
        self.states = defaultdict(list)
        for s in raw_states:
            xcount = np.count_nonzero(s == 2)
            ocount = np.count_nonzero(s == 1)
            self.states[(xcount, ocount)].append(State(s))

        # Init the Q values. The row is state index and column is action index
        # TODO: This nested loop seems nasty. Priority: medium
        self.Q_values = {
            (s, a): 0.0 for v in self.states.values() for s in v for a in self.actions
        }

    def eval(self):
        # eval mode disables epsilon-greedy. Only greedy behaviour.
        self.eval_mode = True

    def train(self):
        # train mode enables epsilon-greedy behaviour.
        self.eval_mode = False

    def find_state(self, s: State) -> Tuple[State | None, str | None]:
        # returns the canonical state, and the transformation that transforms the original state
        # into the canonical state

        xcount = np.count_nonzero(s.s == 2)
        ocount = np.count_nonzero(s.s == 1)
        candidate_states = self.states[(xcount, ocount)]
        board_variations = get_board_variations(s.s, include_self=True)

        # search for board variations in candidate states:
        for t, b in board_variations.items():
            for c in candidate_states:
                if State(b) == c:
                    return c, t

        return None, None

    def get_all_states(self) -> List[State]:
        all_states_flat = []
        for s_list in self.states.values():
            for s in s_list:
                all_states_flat.append(s)

        return all_states_flat

    def discounted_return(self, rewards):
        # This computes discounted sum of future rewards.
        return sum([self.discount_factor**i * r for i, r in enumerate(rewards)])

    def estimate_qvalues(self, buffer: ReplayBuffer, player_mark) -> None:
        # iterate over the replay buffer and update Q values

        for e in buffer.buffer:
            for i, (s, a, _) in enumerate(e):
                if a.mark != player_mark:
                    continue

                future_rewards = [rt[player_mark].r for (_, _, rt) in e[i:]]
                q_estimate = self.discounted_return(future_rewards)

                # The action is on original board. The Q_values are based on canonical board
                # Here we do the transformation
                canonical_state, transformation = self.find_state(s)

                assert (
                    canonical_state and transformation
                ), "couldn't find the canonical state :(("

                # we go from original action to canonical action
                canonical_pos = get_position_transformed(transformation, a.pos)

                l = [ac for ac in self.actions if ac.pos == canonical_pos]
                assert len(l) == 1, "something wrong here!"

                canonical_action = l[0]

                q_oldval = self.Q_values[(canonical_state, canonical_action)]

                self.Q_values[(canonical_state, canonical_action)] = (
                    q_oldval + self.learning_rate * (q_estimate - q_oldval)
                )

    def get_state_values(self, state: State) -> List[Tuple[float, Action]]:
        canonical_state, transformation = self.find_state(state)
        assert (
            canonical_state and transformation
        ), "couldn't find the canonical state. Thats not good bro."

        values = [self.Q_values[(canonical_state, a)] for a in self.actions]
        # Note that here the actions are corresponding to canonical board.
        # We need actions corresponding to the original board
        # From canonical -> original
        actions_transformed = [
            Action(get_position_inverse_transformed(transformation, a.pos), a.mark)
            for a in self.actions
        ]
        return list(zip(values, actions_transformed))

    def __call__(self, state: State) -> Action:
        """returns the best action based on the Q-values stored with 1-epsilon probability"""
        possible_actions = [a for a in self.actions if state.s.reshape(-1)[a.pos] == 0]

        # epsilon-greedy. Take random action with 1-epsilon prob if in train mode
        if not self.eval_mode and np.random.rand() > self.epsilon:
            return possible_actions[np.random.choice(np.arange(len(possible_actions)))]

        state = State(state.s.reshape(3, 3).astype(int))  # type: ignore

        # find the canonical form of the state. TODO: check this work correctly
        canonical_state, transformation = self.find_state(state)

        if not canonical_state or not transformation:
            raise Exception("couldn't find the state, it shouldn't be missing :/")

        possible_actions_canonical = [
            a for a in self.actions if canonical_state.s.reshape(-1)[a.pos] == 0
        ]
        best_action_value = -float("inf")
        best_action_canonical = None
        for a in possible_actions_canonical:
            if self.Q_values[(canonical_state, a)] > best_action_value:
                best_action_canonical = a
                best_action_value = self.Q_values[(canonical_state, a)]

        assert best_action_canonical, "something wrong here!"
        # best_action corresponds to the canonical state. We need the action corresponding to the original state
        # we go from canonical action to board action
        best_pos_original = get_position_inverse_transformed(
            transformation, best_action_canonical.pos
        )
        l = [a for a in possible_actions if a.pos == best_pos_original]
        assert len(l) == 1
        best_action = l[0]

        return best_action
