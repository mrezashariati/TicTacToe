from game_env import Board, Player, GameRunner


def main():
    # Players ready!
    xplayer = Player(mark="X", policy_type="manual")
    oplayer = Player(mark="O", policy_type="random")
    env = Board()
    runner = GameRunner(episodes=1, env=env, players=[xplayer, oplayer])
    episodes = runner.run()
    # turn = xplayer
    # board = Board(do_log=False)
    # print("Game setup complete.")
    # print("Now I'll be commentating the game XOXO")
    # print("The game always start with 'X'")
    # # Lets start the game
    # while not board.finished:
    #     print(f"Player to act: {turn.mark}")
    #     action = turn.play(board.get_game_state())
    #     board.step(action)
    #     turn = xplayer if turn == oplayer else oplayer


if __name__ == "__main__":
    main()
