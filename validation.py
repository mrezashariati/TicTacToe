import logging
from game_env import Board, Player
from entities import State
import numpy as np
from collections import defaultdict
from typing import Dict
import pickle


def lets_see_them_play(xplayer: Player, oplayer: Player):
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

    xplayer.train()
    oplayer.train()


def head_to_head(xplayer: Player, oplayer: Player, episodes=1000):

    assert (
        oplayer.mark == "O" and xplayer.mark == "X"
    ), "Oplayer and Xplayer are not passed correctly"

    oplayer.eval()
    xplayer.eval()

    winners = []

    for _ in range(episodes):
        turn = xplayer
        board = Board()

        while not board.finished:
            action = turn.play(board.get_game_state())
            board.step(action)
            turn = xplayer if turn == oplayer else oplayer

        winner = board._winner if board._winner else "D"
        winners.append(winner)

    oplayer.train()
    xplayer.train()

    values, counts = np.unique_counts(np.array(winners))
    return values.tolist(), np.round(counts / sum(counts) * 100, 2).tolist()


def see_win_lose_patterns(boards: list[Board], outcomes):
    pass


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

    for s in p.get_all_states():
        if Board.is_terminal(s):
            continue

        turn = s.whos_turn()

        # check if the state's player's turn matches the mark of the target player
        if Board._mark_to_number[player.mark] != turn:
            continue

        # TODO: the policy type isn't known here. Fix this. Priority: medium
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

    p = player._policy
    action_values = defaultdict(list)
    for (s, a), v in p.Q_values.items():
        if Board.is_terminal(s) or Board._mark_to_number[player.mark] != s.whos_turn():
            # This will have zero values for all actions. Skipping zeros
            continue

        action_values[a].append(v)

    action_stats = {
        a.pos: (
            float(round(np.quantile(action_values[a], 0.25), 2)),
            float(round(np.median(action_values[a]), 2)),
            float(round(np.quantile(action_values[a], 0.75), 2)),
            float(round(np.min(action_values[a]), 2)),
            float(round(np.max(action_values[a]), 2)),
        )
        for a in action_values.keys()
    }
    return action_stats


def get_state_values(player: Player, s: State) -> Dict[int, float]:
    assert (
        player.policy_type == "rl"
    ), "score distribution across actions is only valid for rl players with state,action value estimations"

    # TODO: the policy type here is not know. Priority: medium
    p = player._policy
    value_action_list = p.get_state_values(s)
    value_list = {i: v for i, (v, _) in enumerate(value_action_list)}
    return value_list


def run_evaluations(oplayer: Player, n_sample_states=5, n_manual_head_to_head=5):
    # Runs this evaluation list:
    # Winrate against Random Policy player
    # state coverage
    # Action score distribution over all states
    # Sample state, action values
    # Head to Head with Manual policy

    # Against Random Policy Player
    win_stats = head_to_head(
        xplayer=Player(mark="X", policy_type="random"), oplayer=oplayer, episodes=3000
    )
    oplayer_winstats = dict(zip(*win_stats))
    logging.info(f"Oplayer winrate against Random Policy Player: {oplayer_winstats}")

    # State Coverage
    state_coverage, _ = get_state_coverage(oplayer, return_sample=False)
    logging.info(
        f"ratio of non-terminal states that has been updated atleast once (state coverage) {state_coverage*100:.2f}%",
    )

    # Action score distribution
    logging.info("Action score distributions 0.25q, 0.5q, 0.75q, min, max:")
    logging.info(
        "\n".join(
            map(
                str,
                sorted(
                    get_score_distribution_across_actions(oplayer).items(),
                    key=lambda x: x[1][2],
                ),
            )
        )
    )

    # See some State, Action values
    with open("./all_valid_O_states.pkl", "rb") as f:
        all_valid_O_states = pickle.load(f)

    for s in np.random.choice(all_valid_O_states, size=n_sample_states, replace=False):
        logging.info(s)
        logging.info(get_state_values(player=oplayer, s=s))

    # Head to Head with Manual policy
    for _ in range(n_manual_head_to_head):
        lets_see_them_play(
            xplayer=Player(mark="X", policy_type="manual"), oplayer=oplayer
        )
