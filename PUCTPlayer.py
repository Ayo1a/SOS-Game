from PUCTNode import PUCTNode
import numpy as np
from constant import *


class PUCTPlayer:
    def __init__(self, c_puct=1.0, simulations=5):
        self.c_puct = c_puct
        self.simulations = simulations
        self.visited_nodes = {}  # מילון לשמירת צמתים

    def play(self, game):
        # יצירת שורש חדש ותחילת חיפושי MCTS
        root = self.get_or_create_node(game)
        for _ in range(self.simulations):
            self.simulate(root)
            root.print_tree()

        # בחירת המהלך הטוב ביותר
        best_action = max(root.children.items(), key=lambda child: child[1].visit_count)[0]
        return best_action

    def get_board_state(self, game):
        """החזרת ייצוג מצב הלוח ללא השחקן הנוכחי"""
        return tuple(tuple(row) for row in game.board)  # ייצוג הלוח כמבנה נתונים (tuple של tuple)

    def get_or_create_node(self, game):
        """מחפש או יוצר צומת חדש"""
        game_state = self.get_board_state(game)  # הוצאת מצב הלוח  # ייצוג מצב המשחק
        if game_state in self.visited_nodes:
            return self.visited_nodes[game_state]  # אם הצומת קיים, מחזירים את הצומת הקיים
        else:
            node = PUCTNode(game)
            self.visited_nodes[game_state] = node
            return node

    def simulate(self, node):
        """Performs a simulation using MCTS"""

        if node.game.game_over:
            return self.evaluate_game(node.game)  # אם המשחק נגמר, מחזירים הערכה

        if not node.children:
            policy, value = self.evaluate_random(node.game)
            node.expand(policy)  # מרחיבים את הצומת עם מהלכים אפשריים

            # בוחרים את המהלך הראשון בהתבסס על ההסתברות הגבוהה ביותר במדיניות
            best_move = max(policy, key=policy.get) if policy else None
            if best_move is None:
                return value  # אם אין מהלך, מחזירים את הערך כפי שהוא

            first_child = node.children[best_move]

            # ביצוע המהלך במשחק
            row, col, letter = best_move
            node.game.make_move(row, col, letter)

            # המשך סימולציה עם הצומת החדש
            value = self.simulate(first_child)

            # שחזור מצב המשחק
            node.game.unmake_move()

            return -value  # החזרת הערך ההפוך כדי להתאים למינימקס

        # שלב הבחירה - בחירת מהלך על פי PUCT
        action, child = node.select(self.c_puct)

        # ביצוע המהלך שנבחר
        row, col, letter = action
        node.game.make_move(row, col, letter)

        # סימולציה על הילד שנבחר
        value = self.simulate(child)

        # שחזור מצב המשחק
        node.game.unmake_move()

        # עדכון הערך בצומת
        child.update(value)

        return -value  # מחזירים את הערך ההפוך

    def evaluate_random(self, game):
        legal_moves = game.legal_moves()
        if not legal_moves:
            return {}, 0.0  # אין מהלכים חוקיים

        move_scores = {}
        current_player = game.current_player
        opponent = PLAYER_1 if game.current_player == PLAYER_2 else PLAYER_2

        for move in legal_moves:
            x, y, letter = move

            # 1️⃣ **תוספת נקודות לשחקן הנוכחי**
            before = game.scores.copy()
            game.make_move(x, y, letter)
            after = game.scores.copy()
            gain = after[current_player] - before[current_player]

            # 2️⃣ **בדוק את כל המהלכים של היריב** לאחר המהלך שלי
            opponent_sos_threat = 0
            for opponent_move in game.legal_moves():
                ox, oy, _ = opponent_move
                game.make_move(ox, oy, 'S' if opponent == 1 else 'O')  # נניח שהיריב שם S או O
                if game.check_sos(ox, oy)[0] > 0:  # אם אחרי המהלך של היריב נוצר SOS
                    opponent_sos_threat = max(opponent_sos_threat,  game.check_sos(ox, oy)[0])
                game.unmake_move()

            # 3️⃣ **מניעת SOS מהיריב**
            opponent_prevention = opponent_sos_threat  # כל סיכון להשלמת SOS של היריב שאנחנו יכולים למנוע

            # ציון סופי - אנחנו נותנים יותר משקל לניקוד שלי, אבל גם למניעת SOS מהיריב
            move_scores[move] = gain * 3 - opponent_prevention * 2  # משקלים - להגדיל נקודות ולמנוע SOS

            game.unmake_move()  # מחזירים את המצב

        # אם כל המהלכים מובילים לאותו ציון, השתמש בהסתברות אחידה
        if all(score == 0 for score in move_scores.values()):
            move_probs = {move: 1 / len(legal_moves) for move in legal_moves}
        else:
            exp_values = np.exp(list(move_scores.values()))
            probabilities = exp_values / np.sum(exp_values)
            move_probs = {move: prob for move, prob in zip(move_scores.keys(), probabilities)}

        # הערך הכללי של המשחק לאחר החישוב
        value = np.tanh(np.mean(list(move_scores.values()))) if move_scores else 0.0

        return move_probs, value


    def evaluate_game(self, game):
        if game.scores[PLAYER_1] > game.scores[PLAYER_2]:
            return 1
        elif game.scores[PLAYER_2] > game.scores[PLAYER_1]:
            return -1
        else:
            return 0

    def evaluate_random_multiple(self, game, num_simulations=10):
        values = []
        for _ in range(num_simulations):
            move_probs, value = self.evaluate_random(game)
            values.append(value)

        mean_value = np.mean(values)
        print(f"Num Simulations: {num_simulations}, Values: {values}, Mean Value: {mean_value}")
        return mean_value
