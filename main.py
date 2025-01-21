import random
from GameDataset import GameDataset
from SOSGame import SOSGame
from PUCTPlayer import PUCTPlayer
from torch import optim, nn
from torch.utils.data import DataLoader

def train_network(network, data, epochs=10, batch_size=64, lr=0.001):
    optimizer = optim.Adam(network.parameters(), lr=lr)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    loader = DataLoader(GameDataset(data), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_policy_loss, total_value_loss = 0, 0
        for state, policy, value in loader:
            optimizer.zero_grad()
            predicted_policy, predicted_value = network(state)
            policy_loss = criterion_policy(predicted_policy, policy)
            value_loss = criterion_value(predicted_value.squeeze(), value)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        print(f"Epoch {epoch + 1}/{epochs}: Policy Loss = {total_policy_loss}, Value Loss = {total_value_loss}")

# Pre-training
def generate_self_play_data(player, games=10000):
    data = []
    for _ in range(games):
        game = SOSGame()
        states, visit_counts, outcomes = [], [], []
        while not game.game_over:
            state = game.encode()
            action = player.play(game)
            visit_count = [node.visit_count for node in player.root.children.values()]
            states.append(state)
            visit_counts.append(visit_count)
            game.make_move(*action)
        outcome = 1 if game.scores[SOSGame.PLAYER_1] > game.scores[SOSGame.PLAYER_2] else -1
        for s, vc in zip(states, visit_counts):
            data.append((s, vc, outcome))
    return data


def main():
    game = SOSGame()
    player = PUCTPlayer(c_puct=1.0, simulations=100)

    while not game.game_over:
        print("Current Board:")
        for row in game.board:
            print(' '.join(row))
        print(f"Scores: {game.scores}")

        if game.current_player == SOSGame.PLAYER_1:
            print("Player 1's turn")
            x, y, letter = player.play(game)
        else:
            print("Player 2's turn")
            legal = game.legal_moves()
            print(f"Legal Moves: {legal}")

            # Randomly decide between S and O, biased towards S
            if random.random() < 0.7:  # 70% chance to choose 'S'
                move = next((move for move in legal if move[2] == 'S'), None)
            else:
                move = next((move for move in legal if move[2] == 'O'), None)

            # Default to the first legal move if specific letter moves are not available
            if not move:
                move = legal[0]

            x, y, letter = move

        print(f"Player {game.current_player} chooses: ({x}, {y}, {letter})")
        game.make_move(x, y, letter)

    print("Game Over!")
    print(f"Final Scores: {game.scores}")


if __name__ == "__main__":
    main()
