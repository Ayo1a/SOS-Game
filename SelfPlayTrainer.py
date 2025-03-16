import random
import torch
import torch.optim as optim


class SelfPlayTrainer:
    def __init__(self, network, simulations=100, replay_size=10000, batch_size=32, learning_rate=1e-3):
        self.network = network
        self.replay_buffer = []
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.simulations = simulations
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    def play_game(self, game):
        """ משחק Self-Play בין שני שחקנים (שני עותקים של הרשת) """
        history = []
        while not game.game_over:
            move_probs, value = self.network.evaluate_network(game)
            legal_moves = game.legal_moves()
            best_move = max(legal_moves, key=lambda move: move_probs[move])
            game.make_move(*best_move)
            history.append((game.encode(), best_move, value))
        return history

    def train(self):
        """ אימון הרשת בעזרת החוויות שנשמרו בזיכרון """
        if len(self.replay_buffer) < self.batch_size:
            return

        # בחר דגימות אקראיות מהזיכרון
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards = zip(*batch)

        # המר את המידע לטנזורים
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        # הפעל את הרשת
        policy, value = self.network(states_tensor)

        # חישוב החישוב עבור ה-policy ו-value
        log_probs = torch.log(policy.gather(1, actions_tensor.unsqueeze(-1)))
        loss_policy = -log_probs * rewards_tensor
        loss_value = (value - rewards_tensor).pow(2)

        # חיבור שתי הפונקציות אובדן
        loss = loss_policy.mean() + loss_value.mean()

        # עדכון משקלים
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def add_to_replay_buffer(self, game_history):
        """ הוסף חוויות לזיכרון """
        for state, action, reward in game_history:
            self.replay_buffer.append((state, action, reward))
            if len(self.replay_buffer) > self.replay_size:
                self.replay_buffer.pop(0)
