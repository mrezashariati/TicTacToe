from typing import Literal
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


# State and Action space of TicTacToe
@dataclass
class State:
    s: NDArray[np.int16]

    def __hash__(self) -> int:
        return hash("".join(map(str, self.s.tolist())))

    def __eq__(self, other) -> bool:
        if isinstance(other, State):
            return bool(np.all(self.s == other.s))

        return False

    def copy(self):
        return State(self.s.copy())


@dataclass
# This class is specific to TicTacToe game
class Action:
    pos: int
    mark: Literal["X", "O"]

    def __hash__(self) -> int:
        return hash(str(self.pos) + self.mark)

    def __eq__(self, other) -> bool:
        if isinstance(other, Action):
            return self.pos == other.pos and self.mark == other.mark

        return False
