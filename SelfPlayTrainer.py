import torch
import os
import numpy as np
from torch import optim
import torch.nn as nn
import pickle
from SOSGame import SOSGame
from constant import *
import copy


class SelfPlayTrainer:
    def __init__(self, player, num_games=10000, data_file='self_play_data.pkl'):
        self.player = player
        self.num_games = num_games
        self.data_file = data_file
        self.data_buffer = []

        # אם קיים קובץ נתונים, נטען אותו
        self.load_data()

    def generate_training_data(self):
        """יצירת נתוני אימון דרך משחק עצמי אם אין נתונים קיימים."""
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

        # שמירת הנתונים שנוצרו לקובץ
        self.save_data()

    def save_data(self):
        """שומר את נתוני המשחקים לקובץ."""
        with open(self.data_file, 'wb') as f:
            pickle.dump(self.data_buffer, f)

    def load_data(self):
        """טוען את נתוני המשחקים מקובץ אם קיים."""
        try:
            with open(self.data_file, 'rb') as f:
                self.data_buffer = pickle.load(f)
            print(f"Loaded {len(self.data_buffer)} games from {self.data_file}")
        except FileNotFoundError:
            print(f"No existing data found, generating new data...")

    def extract_policy(self, root):
        """הפקת התפלגות הסתברויות למדיניות מה-MCTS."""
        total_visits = sum(child.visit_count for child in root.children.values())

        if total_visits == 0:
            legal_moves = root.game.legal_moves()
            if not legal_moves:
                print("Warning: No legal moves available in root state!")
                print("Board:")
                print(root.game.board)
                print("Current player:", root.game.current_player)
                return {}  # or return None, or handle this gracefully in your code
            uniform_prob = 1 / len(legal_moves)
            return {move: uniform_prob for move in legal_moves}

        return {move: child.visit_count / total_visits for move, child in root.children.items()}


class Trainer:
    def __init__(self, network, learning_rate=0.001, batch_size=32, epochs=10, checkpoint_dir='checkpoints'):
        self.network = network
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_policy = nn.CrossEntropyLoss()
        self.loss_value = nn.MSELoss()
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # אתחול של משתנים לאחסון הערכים הטובים ביותר
        self.best_loss = float('inf')
        self.best_epoch = 0

    def save_best_weights(self, epoch):
        """שומר את המשקלים של הרשת אם הם המשקלים הכי טובים."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'best_model_epoch_{epoch}.pth')
        torch.save(self.network.state_dict(), checkpoint_path)
        print(f"Saved best model weights to {checkpoint_path}")

    def load_best_weights(self):
        """מטעין את המשקלים הכי טובים שהיו במהלך האימון."""
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # מיון לפי אפוקות
            latest_checkpoint = checkpoint_files[-1]
            checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
            self.network.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded best model weights from {checkpoint_path}")

    def train(self, data_buffer):
        """
        מאמן את הרשת על הדוגמאות שנאספו.
        כל דוגמה ב-data_buffer היא טפל (state, policies, value):
          - state: תוצאת encode() בגודל [7, BOARD_SIZE, BOARD_SIZE]
          - policies: מילון (dict) שממפה כל פעולה (0 עד 49) להסתברות (כל פעולה = תא בלוח עם 2 אפשרויות)
          - value: הערך (value) של מצב המשחק: 1 = ניצחון, -1 = הפסד, 0 = תיקו
        """

        # חלוקה לשלושה מרכיבים נפרדים מתוך רשימת הדוגמאות
        states, policies, values = zip(*data_buffer)

        # המרת states לטנסור בגודל [batch_size, 7, BOARD_SIZE, BOARD_SIZE]
        states = torch.tensor(np.array(states), dtype=torch.float32)

        # גודל מרחב הפעולות: כל תא בלוח יכול להכיל שתי פעולות (S או O)
        action_space_size = BOARD_SIZE * BOARD_SIZE * 2

        # המרת כל policy למערך בגודל action_space_size עם ערכים מה-dict
        policies = [np.array([p.get(i, 0) for i in range(action_space_size)], dtype=np.float32) for p in policies]
        policies = torch.tensor(np.array(policies), dtype=torch.float32)

        # המרת ערכי value לרשימת טנסורים
        values = torch.tensor(np.array(values), dtype=torch.float32)

        # נאתחל משתנים למעקב אחרי תוצאות
        best_accuracy = 0.0
        best_model_state = None

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            # חיזוי מדיניות וערך מהרשת
            pred_policies, pred_values = self.network(states)

            # שטיחת הפלט של המדיניות למימד [batch_size, action_space_size]
            pred_policies = pred_policies.view(len(policies), action_space_size)

            # חישוב log-softmax למדיניות החזויה לצורך KLDivLoss
            log_pred = torch.nn.functional.log_softmax(pred_policies, dim=1)

            # חישוב הפסד מדיניות (KL divergence)
            loss_p = torch.nn.functional.kl_div(log_pred, policies, reduction='batchmean')

            # חישוב הפסד ערך (MSE או CrossEntropy, תלוי בהגדרת self.loss_value)
            loss_v = self.loss_value(pred_values.squeeze(), values)

            # סך הכל הפסד
            loss = loss_p + loss_v

            # עדכון משקלים
            loss.backward()
            self.optimizer.step()

            # חישוב מדדים להערכה
            total_games = len(values)
            total_wins = sum(1 for v in values if v == 1)  # סה"כ נצחונות
            total_accuracy = sum(
                1 for pv, v in zip(pred_values.squeeze(), values)
                if (pv >= 0.5 and v == 1) or (pv < 0.5 and v == -1)
            )

            # חישוב שיעור ניצחונות ודיוק
            win_rate = total_wins / total_games if total_games > 0 else 0.0
            accuracy = total_accuracy / total_games if total_games > 0 else 0.0

            # הדפסת התקדמות
            print(
                f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.8f}, Win Rate: {win_rate:.2f}, Accuracy: {accuracy:.2f}")

            # אם הדיוק הנוכחי הוא הגבוה ביותר – נשמור את המודל
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = copy.deepcopy(self.network.state_dict())
                print(f"🔥 New best model found at epoch {epoch + 1} with accuracy {accuracy:.2f}")

        # לאחר סיום כל האפוקים – נשמור את המודל הטוב ביותר אם היה שיפור
        if best_model_state is not None:
            torch.save(best_model_state, "checkpoints/best_model.pth")
            print("✅ Best model saved to checkpoints/best_model.pth")
        else:
            print("⚠️ No improvement in accuracy. Model not saved.")
