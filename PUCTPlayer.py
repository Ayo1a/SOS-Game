import heapq

from PUCTNode import *
from SOSGame import *


class PUCTPlayer:
    def __init__(self, c_puct=1.0, network=None, simulations=5):
        self.c_puct = c_puct
        self.simulations = simulations
        self.visited_nodes = {}  # מילון לשמירת צמתים
        self.root = None  # שמירת ה-root בין הסימולציות
        self.state_to_node = {}
        self.network = network

    def play(self, game):
        """ביצוע מהלך בעזרת PUCT ו-MCTS."""
        if self.root is None or self.get_board_state(game) not in self.visited_nodes:
            self.root = self.get_or_create_node(game)  # שמירה על ה-root

        for i in range(self.simulations):
            print(f"Running simulation {i + 1} of {self.simulations}")
            value = self.simulate(self.root)
            self.root.update(value)
        self.root.print_tree()

        # בחירת המהלך הטוב ביותר
        best_action = max(
            self.root.children.items(),
            key=lambda child: (child[1].visit_count, child[1].value)
        )[0]

        # 🔹 **עדכון ה-root אחרי בחירת המהלך**
        if best_action in self.root.children:
            self.root = self.root.children[best_action]

        return best_action

    def get_or_create_node(self, game):
        """מחפש או יוצר צומת חדש."""
        game_state = self.get_board_state(game)
        if game_state in self.state_to_node:
            #print(f"🔄 Using existing node for state: {game_state}")
            return self.state_to_node[game_state]  # מחזיר את הצומת הקיים

        #print(f"🆕 Creating new node for state: {game_state}")
        node = PUCTNode(game)
        self.state_to_node[game_state] = node
        return node

    def get_board_state(self, game):
        """החזרת ייצוג מצב הלוח."""
        return tuple(tuple(row) for row in game.board)

    def simulate(self, node):
        """ביצוע סימולציה מתוך MCTS."""

        # אם המשחק נגמר, מחזירים הערכה סופית של הלוח
        if node.game.game_over:
            return self.evaluate_game(node.game)

        # 🔹 שלב הבחירה: הרחבת צומת חדש או בחירת מהלך קיים
        if not node.children:
            # קבלת מדיניות והערכה מהרשת אם קיימת, אחרת שימוש בהערכה רנדומלית
            policy, value = (
                self.network.predict(node.game.encode())
                if self.network else self.evaluate_random(node.game)
            )

            # הרחבת הצומת עם המדיניות שחושבה
            node.expand(policy, self)

            # שליפת המהלכים החוקיים
            legal_moves = node.game.legal_moves()

            # אם אין מהלכים חוקיים, מחזירים הערכת משחק
            if not legal_moves:
                return self.evaluate_game(node.game)

            # ווידוא שהמדיניות היא מילון תקין
            if isinstance(policy, torch.Tensor):
                policy_np = policy.cpu().numpy()
                policy_dict = {
                    (i, j, piece): policy_np[i, j, idx]
                    for i in range(policy_np.shape[0])
                    for j in range(policy_np.shape[1])
                    for idx, piece in enumerate(['S', 'O'])
                }
                policy = policy_dict

            if not isinstance(policy, dict):
                raise TypeError(f"Expected policy to be a dictionary, but got {type(policy)}")

            # ווידוא שכל המהלכים החוקיים קיימים במדיניות
            missing_moves = [move for move in legal_moves if move not in policy]
            if missing_moves:
                print(f"⚠️ Warning: The following legal moves are missing in policy: {missing_moves}")

            # בחירת המהלך עם ההסתברות הגבוהה ביותר
            max_value = max(policy.get(move, 0.0) for move in legal_moves)
            best_moves = [move for move in legal_moves if policy.get(move, 0.0) == max_value]
            best_move = random.choice(best_moves) if best_moves else random.choice(legal_moves)

        else:
            # 🔹 בחירה לפי PUCT אם לצומת יש ילדים
            best_move, _ = node.select(self.c_puct)

        # 🔹 מציאת הילד המתאים וביצוע סימולציה עליו
        child = node.children.get(best_move)  # שימוש ב-get כדי למנוע KeyError

        if child is None:
            raise ValueError(f"Child node for move {best_move} not found in node.children!")

        value = self.simulate(child)  # המשך הסימולציה

        # 🔹 עדכון הצומת עם הערך שהתקבל
        node.update(-value)

        return -value  # החזרת הערך ההפוך כדי להתאים לאלגוריתם מינימקס

    def evaluate_random(self, game):
        """
        מבצע הערכת מצב חכמה על ידי השוואת המהלך הטוב ביותר של השחקן הנוכחי
        מול האיום הפוטנציאלי הגדול ביותר של היריב.

        מחזיר:
            move_probs (dict): התפלגות הסתברויות על המהלכים החוקיים.
            value (float): הערכת הערך של המצב הנוכחי.
        """
        legal_moves = game.legal_moves()
        if not legal_moves:
            return {}, 0.0  # אין מהלכים חוקיים זמינים

        current_player = game.current_player
        opponent = PLAYER_1 if current_player == PLAYER_2 else PLAYER_2  # זיהוי היריב

        # 1️⃣ **חישוב מראש של כל הרווחים לכל מהלך חוקי**
        move_gains = {}  # מילון שמאחסן את כמות ה-SOS שנוצרה מכל מהלך
        for move in legal_moves:
            x, y, letter = move
            game.make_move(*move)
            move_gains[move] = game.check_sos(x, y)[0]  # מספר צורות SOS שנוצרו במהלך
            game.unmake_move()

        move_scores = {}  # מילון לאחסון הציונים לכל מהלך חוקי

        # 2️⃣ **חישוב ציון לכל מהלך חוקי**
        for move in legal_moves:
            game.make_move(*move)  # לבצע את המהלך
            gain = move_gains[move]  # הרווח המיידי מהמהלך

            # 3️⃣ **מציאת המהלך הכי טוב שנשאר לי לאחר שביצעתי את המהלך הזה**
            remaining_moves = [m for m in legal_moves if (m[0], m[1]) != (move[0], move[1])]
            best_remaining_gain = max(
                (move_gains[m] for m in remaining_moves),
                default=0  # אם אין מהלכים נוספים, הרווח הוא 0
            )

            # 4️⃣ **חישוב הרווח המקסימלי של היריב באזור שהמהלך השפיע עליו**
            affected_positions = game.get_affected_positions(move)
            opponent_best_gain = max(
                (game.check_sos(x, y)[0] for x, y in affected_positions),
                default=0
            )

            game.unmake_move()  # ביטול המהלך כדי לחזור למצב הקודם

            # 5️⃣ **בחירת התרחיש הגרוע ביותר**
            worst_case = max(best_remaining_gain, opponent_best_gain)

            # 6️⃣ **חישוב ציון סופי לכל מהלך**
            w_gain = 10  # משקל לרווח מיידי
            w_opponent = 5  # משקל לרווח הצפוי של היריב
            w_missed = 1  # משקל לרווח שהשחקן הנוכחי עלול להפסיד

            score = (gain * w_gain) - (opponent_best_gain * w_opponent) - (best_remaining_gain * w_missed)
            move_scores[move] = score

        # 7️⃣ **שימוש ב-Softmax לחישוב הסתברויות לכל מהלך**
        if all(score == 0 for score in move_scores.values()):
            move_probs = {move: 1 / len(legal_moves) for move in legal_moves}  # הסתברות אחידה אם כל הציונים 0
        else:
            exp_values = np.exp(list(move_scores.values()))
            probabilities = exp_values / np.sum(exp_values)
            move_probs = {move: prob for move, prob in zip(move_scores.keys(), probabilities)}

        # 8️⃣ **חישוב ערך המצב הכללי**
        value = np.tanh(np.mean(list(move_scores.values()))) if move_scores else 0.0

        return move_probs, value

    def evaluate_game(self, game):
        """הערכת המשחק בהתאם לניצחון."""
        if game.scores[PLAYER_1] > game.scores[PLAYER_2]:
            return 1
        elif game.scores[PLAYER_2] > game.scores[PLAYER_1]:
            return -1
        return 0
