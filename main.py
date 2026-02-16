from game_env import Board, Player, GameRunner
import logging
from argparse import ArgumentParser
import numpy as np


# dev:
from learning import MonteCarloEstimation
from entities import State


def lets_play_it_real(xplayer, oplayer):
    turn = xplayer
    board = Board()

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


# TODO: this func is for manual validation. Move or remove later.
def fetch_state_qvalues(policy: MonteCarloEstimation, state: State):
    cstate = policy.find_state(state)
    assert cstate is not None
    values = [policy.Q_values[(cstate, a)] for a in policy.actions]
    return zip(policy.actions, values)


def main():
    # argument parsing
    parser = ArgumentParser()
    parser.add_argument("--do_log", action="store_true", default=False)
    args = parser.parse_args()
    do_log = bool(args.do_log)

    if do_log:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    xplayer = Player(mark="X", policy_type="random")
    oplayer = Player(mark="O", policy_type="RL")
    env = Board()

    for _ in range(10):
        logging.info(f"generating game episodes...")
        runner = GameRunner(episodes=300, env=env, players=[xplayer, oplayer])
        episodes = runner.run()

        logging.info(
            f"the RL player(s) is(are) learning from {len(episodes)} game episodes!"
        )
        oplayer.learn(episodes)

        win_stats = compare_with_random(oplayer, episodes=200)
        print(win_stats)

    # # Lets test it manually
    # oplayer = Player(mark="O", policy_type="manual")
    # lets_play_it_real(xplayer, oplayer)

    # # test against playing random
    # win_stats = compare_with_random(oplayer, episodes=1000)
    # print(win_stats)

    # # fmt: off
    # mystate = State(np.array(
    #     [[0, 2, 0],
    #      [2, 0, 1],
    #      [0, 2, 1]]
    #     )
    # )
    # print(mystate)
    # p = oplayer._policy
    # out = fetch_state_qvalues(p, mystate)
    # print(sorted(list(out), key=lambda a: a[1], reverse=True))


if __name__ == "__main__":
    main()
