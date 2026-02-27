import logging
from game_env import Board, Player
from learning import MonteCarloEstimation
from entities import State
import numpy as np


def lets_play(xplayer: Player, oplayer: Player):
    turn = xplayer
    board = Board()

    xplayer.eval()
    oplayer.eval()

    logging.info("Game setup complete.")
    logging.info("Now I'll be commentating the game XOXO")
    logging.info("The game always start with 'X'")

    # Lets start the game
    while not board.finished:
        board.pretty_print_board()

        logging.info(f"Player to act: {turn.mark}")
        action = turn.play(board.get_game_state())

        board.step(action)
        logging.info(f"Action carried out.")

        turn = xplayer if turn == oplayer else oplayer

    logging.info(
        f"Game finished. The winner: {board._winner if board._winner else 'Neither. Its an equal.'}",
    )


def compare_with_random(player: Player, episodes=100):
    # random vs random: [D: 12.776, O: 28.654, X: 58.57]
    # rl O (without training, all values are 0.0) vs random X: [D: 5.6, O: 39.66, X: 54.68]

    player.eval()

    winners = []
    if player.mark == "X":
        oplayer = Player(mark="O", policy_type="random")
        xplayer = player
    else:
        xplayer = Player(mark="X", policy_type="random")
        oplayer = player

    for _ in range(episodes):
        turn = xplayer
        board = Board()

        while not board.finished:
            action = turn.play(board.get_game_state())
            board.step(action)
            turn = xplayer if turn == oplayer else oplayer

        winner = board._winner if board._winner else "D"
        winners.append(winner)

    values, counts = np.unique_counts(np.array(winners))
    return values.tolist(), np.round(counts / sum(counts) * 100, 2).tolist()


def get_state_coverage(player: Player, return_sample=False, sample_size=10):
    # The terminal states are discarded, cause one cannot take any actions
    # The states where the next action is the opponents action is also discarded, cause the player cannot take any actions
    # There are some states that the player simply cannot win. It either loses or draws. This states will be always intact as well, cause there will be no rewards for them.
    assert (
        player.policy_type == "rl"
    ), "state coverage is only valid for rl players with state,action value estimations"
    p = player._policy  # type: ignore
    zero_states = []
    n_valid_states = 0
    # TODO: This is not nice, states shouldn't be nested
    for ss in p.states.values():
        for s in ss:
            is_terminal = Board.is_terminal(s)
            if is_terminal:
                continue

            turn = s.whos_turn()
            # TODO: this is ugly bro, there should be a better mapping here between player mark and turn
            if (turn == 2 and player.mark == "O") or (turn == 1 and player.mark == "X"):
                continue

            values = [v[0] for v in p.get_state_values(s)]
            all_zero = np.all(np.array(values) == 0.0)
            if all_zero:
                zero_states.append(s)
            n_valid_states += 1

    sample = None
    if return_sample:
        sample = np.random.choice(zero_states, sample_size, replace=False)

    return 1 - (len(zero_states) / n_valid_states), sample


def get_score_distribution_across_actions(player: Player):
    assert (
        player.policy_type == "rl"
    ), "score distribution across actions is only valid for rl players with state,action value estimations"

    raise NotImplementedError
