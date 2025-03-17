import heapq

from PUCTNode import *
from SOSGame import *


class PUCTPlayer:
    def __init__(self, c_puct=1.0, simulations=5):
        self.c_puct = c_puct
        self.simulations = simulations
        self.visited_nodes = {}  # מילון לשמירת צמתים

    def play(self, game):
        """ביצוע מהלך בעזרת PUCT ו-MCTS."""
        root = self.get_or_create_node(game)
        for i in range(self.simulations):
            print(f"Running simulation {i + 1} of {self.simulations}")
            self.simulate(root)

        # בחירת המהלך הטוב ביותר
        best_action = max(
            root.children.items(),
            key=lambda child: child[1].value / (child[1].visit_count + 1e-6)  # הימנע מחלוקה ב-0
        )[0]

        root.print_tree()
        return best_action

    def get_or_create_node(self, game):
        """מחפש או יוצר צומת חדש."""
        game_state = self.get_board_state(game)
        if game_state in self.visited_nodes:
            return self.visited_nodes[game_state]
        else:
            node = PUCTNode(game)
            self.visited_nodes[game_state] = node
            return node

    def get_board_state(self, game):
        """החזרת ייצוג מצב הלוח."""
        return tuple(tuple(row) for row in game.board)

    def simulate(self, node):
        """ביצוע סימולציה מתוך MCTS."""

        if node.game.game_over:
            return self.evaluate_game(node.game)  # אם המשחק נגמר, מחזירים הערכה

        # שלב הבחירה: האם להרחיב צומת חדש או לבחור מהלך קיים
        if not node.children:
            policy, value = self.evaluate_random(node.game)
            node.expand(policy)  # הרחבת הצומת

            # בדיקה אם הפוליסי ריק
            if not policy or all(val is None for val in policy.values()):
                return self.evaluate_game(node.game)  # אין מהלכים חוקיים

            # מציאת המהלכים הטובים ביותר
            max_value = max(policy.values())
            best_moves = [move for move, val in policy.items() if val == max_value]

            best_move = random.choice(best_moves)
            child = node.children[best_move]  # קישור הצומת לילד החדש
        else:
            best_move, child = node.select(self.c_puct)  # בחירה בעזרת PUCT

        # **ביצוע המהלך**
        row, col, letter = best_move
        node.game.make_move(row, col, letter)

        # עדכון הנתונים של הילד
        child.game = node.game.clone()

        # **הרצת סימולציה על הילד**
        value = self.simulate(child)

        # **שחזור מצב המשחק**
        node.game.unmake_move()

        # **עדכון הצומת**
        child.update(value)

        return -value  # החזרת הערך ההפוך כדי להתאים למינימקס

    def evaluate_random(self, game):
        """ביצוע הערכה חכמה יותר תוך השוואת המהלך הכי טוב שאשאר איתו מול האיום הכי גדול של היריב."""
        legal_moves = game.legal_moves()
        if not legal_moves:
            return {}, 0.0  # אין מהלכים חוקיים

        current_player = game.current_player
        opponent = PLAYER_1 if current_player == PLAYER_2 else PLAYER_2  # זיהוי היריב

        # 1️⃣ **חישוב מראש את כל הרווחים שלי לכל מהלך**
        all_move_gains = {}
        for move in legal_moves:
            x, y, letter = move
            game.make_move(*move)
            all_move_gains[move] = game.check_sos(x, y)[0]  # כמה SOS נוצרו
            game.unmake_move()

        move_scores = {}

        # 2️⃣ **חישוב ציון לכל מהלך חוקי**
        for move in legal_moves:
            game.make_move(*move)  # לבצע מהלך
            gain = all_move_gains[move]  # הרווח שלי מהמהלך

            # 3️⃣ **מציאת המהלך הכי טוב שנשאר לי לאחר שעשיתי את המהלך הזה**
            remaining_moves = [
                m for m in legal_moves if m != move
            ]
            remaining_gains = [all_move_gains[m] for m in remaining_moves]
            best_remaining_gain = max(remaining_gains) if remaining_gains else 0  # אם אין מהלכים, 0

            # 4️⃣ **חישוב רווח מקסימלי של היריב – רק באזור המושפע**
            affected_positions = game.get_affected_positions(move)  # רשימה של תאים שהמהלך השפיע עליהם
            opponent_best_gain = max(
                (game.check_sos(x, y)[0] for x, y in affected_positions),
                default=0
            )

            game.unmake_move()  # לבטל את המהלך ולחזור למצב קודם

            # 5️⃣ **בחירת הגורם החזק יותר לפגיעה בציון שלנו**
            worst_case = max(best_remaining_gain, opponent_best_gain)

            # 6️⃣ **חישוב ציון סופי**
            move_scores[move] = (gain * 3) - (worst_case * 2)  # שיפור חסימת SOS

        # 7️⃣ **חישוב הסתברות לפי Softmax**
        if all(score == 0 for score in move_scores.values()):
            move_probs = {move: 1 / len(legal_moves) for move in legal_moves}  # הסתברות אחידה
        else:
            exp_values = np.exp(list(move_scores.values()))
            probabilities = exp_values / np.sum(exp_values)
            move_probs = {move: prob for move, prob in zip(move_scores.keys(), probabilities)}

        # 8️⃣ **חישוב ערך כללי של המצב**
        value = np.tanh(np.mean(list(move_scores.values()))) if move_scores else 0.0

        return move_probs, value

    def evaluate_game(self, game):
        """הערכת המשחק בהתאם לניצחון."""
        if game.scores[PLAYER_1] > game.scores[PLAYER_2]:
            return 1
        elif game.scores[PLAYER_2] > game.scores[PLAYER_1]:
            return -1
        return 0
