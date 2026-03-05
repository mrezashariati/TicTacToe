from game_env import Board, Player, GameRunner
import logging
from argparse import ArgumentParser
import numpy as np
from validation import (
    head_to_head,
    get_state_coverage,
    lets_see_them_play,
    get_score_distribution_across_actions,
    get_state_values,
)
from entities import State

import pickle


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

    for _ in range(5):
        # Generate episodes
        logging.info(f"generating game episodes...")
        runner = GameRunner(num_episodes=4_000, env=Board, players=[xplayer, oplayer])
        runner.run()
        episodes = runner.get_generated_episodes()
        winners = runner.get_winners()
        counts, values = np.unique_counts(winners)
        print(
            "generated episodes winner stats:",
            list(zip(map(str, counts), map(int, values))),
        )

        # Learn
        logging.info(
            f"the RL player(s) is(are) learning from {len(episodes)} game episodes!"
        )
        oplayer.learn(episodes)
        # xplayer.learn(episodes)

        # compare with random policy
        win_stats = head_to_head(
            xplayer=Player(mark="X", policy_type="random"), oplayer=oplayer
        )
        print("head to head winrate stats:", win_stats)

    state_coverage, _ = get_state_coverage(oplayer, return_sample=False)
    print(
        f"ratio of non-terminal states that has been updated atleast once (state coverage) {state_coverage*100:.2f}%",
    )

    print("Action score distributions 0.25q, 0.5q, 0.75q, min, max:")
    print(
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

    with open("./all_valid_O_states.pkl", "rb") as f:
        all_valid_O_states = pickle.load(f)

    for s in np.random.choice(all_valid_O_states, size=5, replace=False):
        print(s)
        print(get_state_values(player=oplayer, s=s))


if __name__ == "__main__":
    main()
