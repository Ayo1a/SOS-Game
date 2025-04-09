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

#1. Debugging and Preliminary Training

def smart_rollouts(game, player, simulations=100):
    """ ביצוע rollouts חכמים בסימולציות עבור MCTS """
    rollouts = []
    for _ in range(simulations):
        temp_game = game.copy()  # שיחזור מצב המשחק
        while not temp_game.game_over:
            legal_moves = temp_game.legal_moves()
            move = random.choice(legal_moves)  # במקרה זה דגימה רנדומלית
            temp_game.make_move(*move)
        rollouts.append(temp_game.get_winner())  # שמירה על תוצאות
    return np.mean(rollouts)  # חזרה על ממוצע תוצאות ה-rollouts

# 2. Pre-train the Network

def pre_train_network(num_games=10000, model_path="sos_weights.pth"):
    """ מאמן את הרשת על ידי משחק עצמי בעזרת MCTS """
    # יצירת רשת נוירונים
    network = GameNetwork()

    try:
        network.load_state_dict(torch.load(model_path))
        print(f"Loaded existing model weights from {model_path}.")
    except FileNotFoundError:
        print("No existing model found. Training from scratch.")

    # יצירת שחקן MCTS
    player = PUCTPlayer(c_puct=10, network=network, simulations=30)

    # יצירת נתוני אימון דרך משחק עצמי
    self_play_trainer = SelfPlayTrainer(player, num_games=num_games)
    self_play_trainer.generate_training_data()

    # אימון הרשת
    trainer = Trainer(network, epochs=50)
    trainer.train(self_play_trainer.data_buffer)

    # שמירת המשקלים
    torch.save(network.state_dict(), model_path)
    print(f"Model saved to {model_path}.")

# 3. Running the Training Loop for PUCT Agent

def train_puct_agent(num_games=10000, epochs=50, model_path="sos_weights.pth"):
    """ מאמן את שחקן ה-PUCT """
    # יצירת רשת נוירונים
    network = GameNetwork()

    try:
        network.load_state_dict(torch.load(model_path))
        print(f"Loaded existing model weights from {model_path}.")
    except FileNotFoundError:
        print("No existing model found. Training from scratch.")

    # יצירת שחקן PUCT
    player = PUCTPlayer(c_puct=10, network=network, simulations=30)

    # יצירת נתוני אימון דרך משחק עצמי
    self_play_trainer = SelfPlayTrainer(player, num_games=num_games)
    self_play_trainer.generate_training_data()

    # אימון הרשת
    trainer = Trainer(network, epochs=epochs)
    trainer.train(self_play_trainer.data_buffer)

    # שמירת המשקלים
    torch.save(network.state_dict(), model_path)
    print(f"Model saved to {model_path}.")

# 4. Evaluating the Agent with ELO

def update_elo(ra, rb, result, k=32):
    """ חישוב דירוג ELO """
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    new_ra = ra + k * (result - ea)
    return new_ra

def evaluate_agent(agent1, agent2, num_games=100):
    """ הערכת ביצועים בין שני שחקנים עם מערכת דירוג ELO """
    elo_A = 1500  # דירוג התחלתי
    elo_B = 1500

    for _ in range(num_games):
        game = SOSGame()
        while not game.game_over:
            move1 = agent1.play(game)
            game.make_move(*move1)
            if game.game_over:
                break
            move2 = agent2.play(game)
            game.make_move(*move2)

        winner = game.get_winner()
        if winner == 1:
            elo_A = update_elo(elo_A, elo_B, 1)
            elo_B = update_elo(elo_B, elo_A, 0)
        elif winner == 2:
            elo_A = update_elo(elo_A, elo_B, 0)
            elo_B = update_elo(elo_B, elo_A, 1)

    print(f"ELO Rating after {num_games} games: Agent1 = {elo_A}, Agent2 = {elo_B}")


# 5. Main Function to Run the Full Pipeline

def train_network(num_games=1000, epochs=50, model_path="sos_weights.pth"):
    # אימון ראשוני של הרשת
    pre_train_network(num_games=num_games, model_path=model_path)

    # אימון של שחקן PUCT
    train_puct_agent(num_games=num_games, epochs=epochs, model_path=model_path)

    # הערכת ביצועים בין גרסאות של השחקן
    agent1 = PUCTPlayer(c_puct=10, network=GameNetwork())
    agent2 = PUCTPlayer(c_puct=10, network=GameNetwork())
    evaluate_agent(agent1, agent2, num_games=100)


if __name__ == "__main__":
    train_network(num_games=1000, epochs=10, model_path="sos_weights.pth")
    #play_game()