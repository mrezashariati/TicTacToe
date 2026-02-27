from game_env import Board, Player, GameRunner
import logging
from argparse import ArgumentParser
import numpy as np
from validation import compare_with_random, get_state_coverage, lets_play
import time

# dev:
from learning import MonteCarloEstimation
from entities import State


def main():
    # argument parsing
    parser = ArgumentParser()
    parser.add_argument("--do_log", action="store_true", default=False)
    args = parser.parse_args()
    do_log = bool(args.do_log)

    if do_log:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    xplayer = Player(mark="X", policy_type="random")
    oplayer = Player(mark="O", policy_type="rl")

    for _ in range(2):
        # Generate episodes
        logging.info(f"generating game episodes...")
        runner = GameRunner(num_episodes=2_500, env=Board, players=[xplayer, oplayer])
        runner.run()
        episodes = runner.get_generated_episodes()

        # Learn
        logging.info(
            f"the RL player(s) is(are) learning from {len(episodes)} game episodes!"
        )
        oplayer.learn(episodes)

        # compare with random policy
        win_stats = compare_with_random(oplayer, episodes=1000)
        print(win_stats)

    state_coverage, _ = get_state_coverage(oplayer)
    print(
        f"ratio of non-terminal states that has been updated atleast once {state_coverage*100:.2f}%",
    )


if __name__ == "__main__":
    main()
