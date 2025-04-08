import torch

from GameNetwork import GameNetwork
from SOSGame import SOSGame
from PUCTPlayer import PUCTPlayer  # מחלקת השחקן המבוסס על MCTS
from SelfPlayTrainer import Trainer, SelfPlayTrainer
from constant import *
import numpy as np
import random

# Game Loop
def play_game():
    """ריצה של המשחק עם השחקן האנושי והמחשב (עכשיו עם PUCT)."""
    game = SOSGame()
    game.set(board= np.array([
        [' ', 'O', 'O', ' ', ' '],
        ['S', 'S', ' ', 'S', 'O'],
        ['S', ' ', 'O', 'O', ' '],
        ['O', 'S', 'O', 'S', ' '],
        [' ', ' ', ' ', 'O', 'O']
    ], dtype=str), current_player=PLAYER_2, scores={PLAYER_1: 3, PLAYER_2: 0})

    puct_player = PUCTPlayer(c_puct=10.0, simulations=1) #c_puct = explotation/exploration balance

    while not game.game_over:
        game.display_board()
        print(f"Player {game.current_player}'s turn")

        if game.current_player == PLAYER_1:
            while True:
                try:
                    move = input("Enter move (row col letter): ").split()
                    if len(move) != 3:
                        raise ValueError("Invalid format. Use: row col letter")
                    x, y, letter = int(move[0]), int(move[1]), move[2].lower()
                    game.make_move(x, y, letter)
                    break
                except Exception as e:
                    print(f"Invalid move: {e}. Try again.")
        else:
            print("Computer's turn...")
            move = puct_player.play(game)
            x, y, letter = move
            print(f"move = ({x}, {y}, {letter})")
            game.make_move(x, y, letter)

        print(f"Current scores - Player 1: {game.scores[PLAYER_1]}, Player 2: {game.scores[PLAYER_2]}")

    game.display_board()
    print(game.status())


def train_network(num_games=10000, epochs=100, model_path="sos_weights.pth"):
    """ מאמן את הרשת באמצעות משחק עצמי ושומר את המשקלים """

    # יצירת רשת נוירונים
    network = GameNetwork()

    # טעינת משקלים קיימים אם קיימים
    try:
        network.load_state_dict(torch.load(model_path))
        print(f"Loaded existing model weights from {model_path}.")
    except FileNotFoundError:
        print("No existing model found. Training from scratch.")

    # יצירת שחקן עם PUCT
    player = PUCTPlayer(c_puct=10, network=network, simulations=30)

    # יצירת נתוני אימון דרך משחק עצמי
    self_play_trainer = SelfPlayTrainer(player, num_games=num_games)

    # אם יש כבר נתונים קודמים, נטעין אותם
    #if not self_play_trainer.data_buffer:
    self_play_trainer.generate_training_data()  # יצירת נתוני אימון אם לא קיימים

    # יצירת מאמן עבור הרשת
    trainer = Trainer(network, epochs=epochs)

    # טעינת המשקלים הכי טובים אם יש
    trainer.load_best_weights()

    # אימון הרשת על הנתונים שנאספו
    trainer.train(self_play_trainer.data_buffer)

    # שמירת המשקלים בסיום האימון
    torch.save(network.state_dict(), model_path)
    print(f"Model saved to {model_path}.")


if __name__ == "__main__":
    train_network(num_games=100, epochs=50, model_path="sos_weights.pth")
    #play_game()