from game_env import Board, Player


def main():
    # Players ready!
    xplayer = Player(mark="X", policy_type="manual")
    oplayer = Player(mark="O", policy_type="RL")
    board = Board(players=[oplayer, xplayer])
    print("Game setup complete.")
    print("Now I'll be commentating the game XOXO")
    # Lets start the game
    while not board.finished:
        pos = board.players[board.turn].play(board.get_game_state())
        board.put_mark(pos=pos)


if __name__ == "__main__":
    main()
