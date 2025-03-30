import numpy as np
import torch
from torch import optim
import torch.nn as nn
from SOSGame import SOSGame
from constant import *

class SelfPlayTrainer:
    def __init__(self, player, num_games=1000):
        self.player = player
        self.num_games = num_games
        self.data_buffer = []

    def generate_training_data(self):
        """יצירת נתוני אימון דרך משחק עצמי."""
        for _ in range(self.num_games):
            game = SOSGame()
            game_data = []
            
            while not game.game_over:
                state = game.encode()
                move = self.player.play(game)
                policy = self.extract_policy(self.player.root)
                game_data.append((state, policy))
                game.make_move(*move)
            
            winner = self.player.evaluate_game(game)
            for state, policy in game_data:
                self.data_buffer.append((state, policy, winner))

    def extract_policy(self, root):
        """הפקת התפלגות הסתברויות למדיניות מה-MCTS."""
        total_visits = sum(child.visit_count for child in root.children.values())
        return {move: child.visit_count / total_visits for move, child in root.children.items()}

class Trainer:
    def __init__(self, network, learning_rate=0.001, batch_size=32, epochs=10):
        self.network = network
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_policy = nn.CrossEntropyLoss()
        self.loss_value = nn.MSELoss()

    def train(self, data_buffer):
        """
        מאמן את הרשת על הדוגמאות שנאספו.
        כל דוגמה ב-data_buffer היא טפל (state, policies, value):
          - state: תוצאת encode() בגודל [7, BOARD_SIZE, BOARD_SIZE]
          - policies: מילון (dict) שממפה כל פעולה (0 עד 49) להסתברות (כל פעולה = תא בלוח עם 2 אפשרויות)
          - value: הערך (value) של מצב המשחק
        """
        # חלוקה לשלושה מרכיבים: states, policies, values
        states, policies, values = zip(*data_buffer)

        # המרת states לטנסור PyTorch; הפלט יהיה בגודל [batch_size, 7, BOARD_SIZE, BOARD_SIZE]
        states = torch.tensor(np.array(states), dtype=torch.float32)

        # כעת נגדיר את גודל החלל הפעולות: עבור לוח של BOARD_SIZE x BOARD_SIZE עם 2 אפשרויות לכל תא
        action_space_size = BOARD_SIZE * BOARD_SIZE * 2  # לדוגמה, אם BOARD_SIZE = 5, אז 5*5*2 = 50

        # המרת policies ממילונים למערך one-hot בגודל [batch_size, action_space_size]
        # כל מילון צריך לספק ערכים עבור 0 עד action_space_size-1 (כלומר, 0 עד 49)
        policies = [np.array([p.get(i, 0) for i in range(action_space_size)], dtype=np.float32) for p in policies]
        policies = np.array(policies, dtype=np.float32)
        policies = torch.tensor(policies, dtype=torch.float32)

        # המרת values לטנסור
        values = torch.tensor(np.array(values), dtype=torch.float32)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            # קבלת הפלט מהרשת:
            # pred_policies מגיע בצורה [batch_size, BOARD_SIZE, BOARD_SIZE, 2] כלומר [batch, 5, 5, 2]
            pred_policies, pred_values = self.network(states)

            # שטיחת הפלט של המדיניות: הפיכת pred_policies ל-[batch_size, 50]
            pred_policies = pred_policies.view(len(policies), action_space_size)

            # לעבודה עם KLDivLoss, אנו צריכים לחשב log_softmax על הפלט
            log_pred = torch.nn.functional.log_softmax(pred_policies, dim=1)

            # loss_policy: הפסד המבוסס על KL divergence בין התפלגות הפלט לבין ה-target (policies)
            # כאן נניח ש-target policies הם התפלגות (one-hot או חלוקה אחרת) בגודל [batch_size, 50]
            loss_p = torch.nn.functional.kl_div(log_pred, policies, reduction='batchmean')

            # loss_value: הפסד עבור חיזוי הערך
            loss_v = self.loss_value(pred_values.squeeze(), values)

            loss = loss_p + loss_v

            loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item()}")
