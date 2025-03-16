from SOSGame import SOSGame
from PUCTPlayer import PUCTPlayer  # מחלקת השחקן המבוסס על MCTS
from constant import *
import numpy as np
import random

def print_board(game):
    """ פונקציה להדפסת לוח המשחק בצורה קריאה """
    print("\n    " + " ".join(str(i) for i in range(BOARD_SIZE)))  # מספרי עמודות
    print("  " + "-" * 17)
    for i in range(BOARD_SIZE):
        row = " ".join(game.board[i, :])
        print(f"{i} | {row} |")
    print("  " + "-" * 17)
    print(f"Score - Player: {game.scores[PLAYER_1]}, Computer: {game.scores[PLAYER_2]}\n")


def get_player_move(game):
    """ מבקש מהשחקן לבחור מהלך """
    while True:
        try:
            move = input("Enter your move as 'row col letter' (e.g., 2 3 S): ").split()
            if len(move) != 3:
                raise ValueError("Invalid input format")

            x, y, letter = int(move[0]), int(move[1]), move[2].upper()
            if (x, y, letter) not in game.legal_moves():
                raise ValueError("Invalid move, try again.")

            return x, y, letter
        except ValueError as e:
            print(f"Error: {e}")


def play_human_vs_ai():
    """ משחק SOS בין שחקן אנושי למחשב """
    game = SOSGame()
    ai_player = PUCTPlayer(c_puct=1.0, simulations=10)  # המחשב

    print("Welcome to SOS Game! You are Player 1.")
    print_board(game)

    while not game.game_over:
        if game.current_player == PLAYER_1:
            print("Your Turn:")
            move = get_player_move(game)  # קלט מהשחקן
        else:
            print("Computer is thinking...")
            move = ai_player.play(game)  # המחשב בוחר מהלך

        game.make_move(*move)
        print_board(game)

    # סיום המשחק
    print("Game Over!")
    print(f"Final Score - Player: {game.scores[PLAYER_1]}, Computer: {game.scores[PLAYER_2]}")

    if game.scores[PLAYER_1] > game.scores[PLAYER_2]:
        print("You win! 🎉")
    elif game.scores[PLAYER_2] > game.scores[PLAYER_1]:
        print("Computer wins! 🤖")
    else:
        print("It's a draw!")

def generate_random_board(board_size=5, fill_prob=0.5):
    """
    Creates a random SOS board.
    - board_size: the size of the board (NxN)
    - fill_prob: probability of placing 'S' or 'O' instead of an empty space
    """
    choices = ['S', 'O', ' ']
    probabilities = [fill_prob / 2, fill_prob / 2, 1 - fill_prob]  # רוב הסיכויים שהמשבצת תהיה ריקה
    board = np.random.choice(choices, size=(board_size, board_size), p=probabilities)
    return board


def distribute_sos_score(game = SOSGame()):
    total_sos = 0

    seen_sos = set()

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            _, sos_positions = game.check_sos(x, y)  # מחזיר רשימה של קואורדינטות SOS

            for sos in sos_positions:
                sos_tuple = tuple(sorted(sos))  # מוודא שהסדר לא משנה
                seen_sos.add(sos_tuple)

    total_sos = len(seen_sos)

    if total_sos == 0:
        return  # אין מה לחלק אם אין SOS

    # מחלקים את הניקוד רנדומלית בין השחקנים
    player_1_score = np.random.binomial(total_sos, 0.5)  # השחקן הראשון מקבל בערך חצי
    player_2_score = total_sos - player_1_score  # השאר הולך לשחקן השני

    game.scores[PLAYER_1] = player_1_score
    game.scores[PLAYER_2] = player_2_score

    print(f"Total SOS: {total_sos}, Player 1: {player_1_score}, Player 2: {player_2_score}")

if __name__ == "__main__":
    game = SOSGame()

    game.board = np.array([
        [' ', 'O', 'O', ' ', ' '],
        ['S', 'S', ' ', 'S', 'O'],
        ['S', ' ', 'O', 'O', ' '],
        ['O', 'S', 'O', 'S', ' '],
        [' ', ' ', ' ', 'O', 'O']
    ], dtype=str)

    distribute_sos_score(game)
    print(f"player = {game.current_player}")

    ai_player = PUCTPlayer(c_puct=1.0, simulations=10)  # המחשב
    ai_player.play(game)

"""
    # יצירת 100 משחקים רנדומליים
    num_games = 100
    game_states = [generate_random_board(board_size=5) for _ in range(num_games)]

    # הרצת evaluate_random_multiple על כל משחק
    ai_player = PUCTPlayer(c_puct=1.0, simulations=1)  # יצירת שחקן מחשב

    results = []
    for i, board in enumerate(game_states):
        print(f"\nGame {i + 1}:")
        print(board)

        game.board = board  # עדכון לוח המשחק
        distribute_sos_score(game)
        print(f"player 1 score: {game.scores[PLAYER_1]}, player 2 score: {game.scores[PLAYER_2]}")

        mean_value = ai_player.evaluate_random_multiple(game, num_simulations=1)
        results.append(mean_value)

    # הדפסת הממוצע של כל המשחקים
    print("\nFinal Results:")
    print(results)"""

    #ai_player.evaluate_random_multiple(game, num_simulations=20)

    #play_human_vs_ai()  # הרצת המשחק
