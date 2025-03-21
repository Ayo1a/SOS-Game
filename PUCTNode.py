import math
import random
import numpy as np
from constant import *

class PUCTNode:
    def __init__(self, game, action=None, prior=0):
        self.game = game
        self.action = action  # הפעולה שהובילה לצומת הזה
        self.children = {}  # צמתים ילדים
        self.visit_count = 0
        self.value = 0
        self.is_fully_expanded = False
        self.untried_actions = game.legal_moves()  # מהלכים שעדיין לא נוסו
        self.prior = prior  # הסתברות ראשונית של המהלך (P)

    def set_game_and_clear_previous_data(self, new_game):
        """ Update the node with a new game state """
        self.game = new_game
        self.untried_actions = new_game.legal_moves()  # Recalculate legal moves
        self.children.clear()  # Remove previous children if state has changed
        self.visit_count = 0
        self.total_value = 0  # Reset accumulated values

    def select(self, c_puct):
        """בחר פעולה על פי Upper Confidence Bound for Trees (PUCT)."""
        best_actions = []
        best_ucb = -float('inf')

        for action, child_node in self.children.items():
            x, y, letter = action
            if not self.game.is_valid(x, y) or self.game.board[x, y] != ' ':
                continue  # מדלגים על מהלכים לא חוקיים

            # חישוב Q-value
            q_value = child_node.value / (child_node.visit_count + 1)

            # חישוב רכיב ה-exploration
            exploration_term = c_puct * child_node.prior * math.sqrt(math.log(self.visit_count) / (child_node.visit_count + 1))

            # חישוב UCB
            ucb = q_value + exploration_term
            #print(f"select Action: {action}, Q: {q_value:.2f}, UCB: {ucb:.2f}, Visits: {child_node.visit_count}")

            if ucb > best_ucb:
                best_ucb = ucb
                best_actions = [(action, child_node)]
            elif ucb == best_ucb:
                best_actions.append((action, child_node))

        return random.choice(best_actions)

    def expand(self, policy, player):
        """הרחבת הצומת תוך שימוש ב- get_or_create_node כדי למנוע כפילות"""
        for move in self.untried_actions:
            if move not in self.children:
                # מבצעים את המהלך על המשחק הנוכחי
                self.game.make_move(*move)

                # 🔹 במקום ליצור PUCTNode חדש, משתמשים ב- get_or_create_node
                child_node = player.get_or_create_node(self.game.clone())

                # עדכון הורה ופרטי המהלך
                child_node.action = move
                child_node.prior = policy.get(move, 0)

                self.children[move] = child_node  # הוספת הצומת לילדים

                # שחזור המהלך (כי self.game הוא עותק חי!)
                self.game.unmake_move()

        self.untried_actions = []  # כל המהלכים נוסו, הצומת מורחב במלואו
        self.is_fully_expanded = True

        #print(f"Expanding node: {self.action}, new children: {len(self.children)}, {self.children}")

    def update(self, value):
        """עדכון ערך הצומת לאחר סימולציה."""
        self.value = (self.value * self.visit_count + value) / (self.visit_count + 1)
        self.visit_count += 1

        #print(f"Node ID: {id(self)} Action: {self.action}| Visits: {self.visit_count}, Value: {self.value:.2f}")

    def __str__(self):
        """Return a readable string representation of the node."""
        return (f"Node ID: {id(self)}, Action: {self.action}, Visits: {self.visit_count}, "
                f"Value: {self.value:.2f}, Children: {len(self.children)}, Prior: {self.prior}")

    def __repr__(self):
        """Return a detailed string representation for debugging."""
        return self.__str__()

    def print_tree(self, indent=0):
        """פונקציה להדפסת העץ בצורה היררכית"""
        # הדפסת המידע של הצומת הנוכחי
        if self.visit_count > 0:
            print(" " * indent + str(self))

        # הדפסת כל הילדים של הצומת
        for action, child in self.children.items():
            child.print_tree(indent + 2)  # הוספת רווחים עבור רמת ההיררכיה

