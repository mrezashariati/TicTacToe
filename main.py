from game_env import Board, Player, GameRunner
import logging
from argparse import ArgumentParser
import numpy as np
from validation import head_to_head, run_evaluations
import copy
from datetime import datetime


def main():
    # argument parsing
    parser = ArgumentParser()
    parser.add_argument("--do_log", action="store_true", default=False)
    parser.add_argument("--oplayer-load-path", required=False)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    do_log = bool(args.do_log)
    eval = bool(args.eval)
    oplayer_load_path = args.oplayer_load_path

    if do_log:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if eval and oplayer_load_path:
        # Load the xplayer and run the evaluation
        oplayer = Player.load(oplayer_load_path)

        run_evaluations(oplayer=oplayer)
        return

    if not eval:
        # Else, train, save and then validate
        # xplayer = Player(
        #     mark="X",
        #     policy_type="rl",
        #     policy_config={
        #         "learning_rate": 0.5,
        #         "discount_factor": 0.9,
        #         "epsilon": 0.6,
        #     },
        # )
        xplayer = Player(
            mark="X",
            policy_type="random",
        )
        oplayer = Player(
            mark="O",
            policy_type="rl",
            policy_config={
                "learning_rate": 0.5,
                "discount_factor": 0.5,
                "epsilon": 0.6,
            },
        )

        best_oplayer = oplayer
        best_oplayer_winrate = 0
        for i in range(12_000):
            # Generate episodes
            # logging.info(f"generating game episodes...")
            runner = GameRunner(num_episodes=1, env=Board, players=[xplayer, oplayer])
            runner.run()
            episodes = runner.get_generated_episodes()
            winners = runner.get_winners()
            counts, values = np.unique_counts(winners)

            # Oplayer learns at each iteration
            oplayer.learn(episodes)

            # # Xplayer learns periodically
            # if i % 2 == 0:
            #     xplayer.learn(episodes)

            if i % 200 == 0:
                # compare with random policy
                win_stats = head_to_head(
                    xplayer=Player(mark="X", policy_type="random"), oplayer=oplayer
                )
                win_stats = dict(zip(*win_stats))

                # keep the player if best
                oplayer_winrate = win_stats["O"]
                if oplayer_winrate > best_oplayer_winrate:
                    best_oplayer_winrate = oplayer_winrate
                    best_oplayer = copy.deepcopy(oplayer)

                logging.info(f"{i} iteration: head to head winrate stats: {win_stats}")

        # Save the player
        best_oplayer.save(
            f"./players/oplayer/{datetime.now().strftime('%d_%m_%Y_%H:%M%:%S')}"
        )

        # Evaluation
        run_evaluations(oplayer=best_oplayer)

    else:
        raise Exception("Not a valid combo of run args!")


if __name__ == "__main__":
    main()
