import random
import numpy as np
from constant import *
import copy
import os
import torch
from PUCTPlayer import *

class SOSGame:

    def __init__(self):
        """Initializes the game board and player settings."""
        self.board = np.full((BOARD_SIZE, BOARD_SIZE), ' ', dtype=str)
        self.current_player =PLAYER_1
        self.scores = {PLAYER_1: 0, PLAYER_2: 0}
        self.game_over = False
        self.move_history = []

    def set(self, board = np.full((BOARD_SIZE, BOARD_SIZE), ' ', dtype=str), current_player = PLAYER_1,  scores = {PLAYER_1: 0, PLAYER_2: 0}):
        self.board = board
        self.current_player = current_player
        self.scores = scores

    def make_move(self, x, y, letter):
        """Executes a move and updates the score. The turn always switches after a move."""
        if not self.is_valid(x, y) or self.board[x, y] != ' ' or letter.upper() not in ('S', 'O'):
            raise ValueError(f"Invalid move {self.board[x, y]}, {letter}")

        letter = letter.upper()
        self.board[x, y] = letter
        points_scored, _ = self.check_sos(x, y)
        self.scores[self.current_player] += points_scored
        self.move_history.append((x, y, letter, self.current_player, points_scored))

        if self.check_game_over():
            self.game_over = True
        else:
            self.current_player = PLAYER_1 if self.current_player == PLAYER_2 else PLAYER_2

    def unmake_move(self):
        """Reverts the board to the previous state by undoing the last move."""
        if not self.move_history:
            return

        # שחזור המידע מתוך היסטוריית המהלכים
        x, y, letter, previous_player, points_scored = self.move_history.pop()

        # שחזור הלוח והניקוד
        self.board[x, y] = ' '  # מחזירים את התו למצבו הריק
        self.scores[previous_player] -= points_scored  # מורידים את הניקוד של השחקן שביצע את המהלך

        # שחזור השחקן בתור
        self.current_player = previous_player

        # בדוק אם המשחק הסתיים ושחזר אותו
        self.game_over = False


    def clone(self):
        """Creates a deep copy of the current game state."""
        cloned_game = self.__class__()  # Supports subclassing
        cloned_game.__dict__ = copy.deepcopy(self.__dict__)  # Deep copy of all attributes
        return cloned_game

    def encode(self):
        """
        מקודד את מצב המשחק כמטריצת one-hot עם מידע נוסף.
        הפלט הוא מערך בגודל [7, BOARD_SIZE, BOARD_SIZE] כאשר:
          - הערוצים 0-2: one-hot encoding של הלוח עבור 'S', 'O' ו־' '.
          - הערוצים 3-6: מידע נוסף (מי השחקן הנוכחי, ניקוד לשחקן 1, ניקוד לשחקן 2, האם המשחק נגמר)
        """
        # קידוד one-hot של הלוח (3 ערוצים)
        encoded_board = np.stack([
            (self.board == 'S').astype(np.float32),
            (self.board == 'O').astype(np.float32),
            (self.board == ' ').astype(np.float32)
        ])

        # מידע נוסף: current_player, score of PLAYER_1, score of PLAYER_2, game_over (0 או 1)
        extra_info = np.array([self.current_player - 1, self.scores[1], self.scores[2], self.game_over],
                              dtype=np.float32)

        # הרחבת extra_info למימדים תואמים ללוח: [4, BOARD_SIZE, BOARD_SIZE]
        extra_info = np.repeat(extra_info[:, np.newaxis, np.newaxis], BOARD_SIZE, axis=1)
        extra_info = np.repeat(extra_info, BOARD_SIZE, axis=2)

        # חיבור הלוח עם המידע הנוסף – מקבלים מערך בגודל [7, BOARD_SIZE, BOARD_SIZE]
        encoded_state = np.concatenate([encoded_board, extra_info], axis=0)
        return encoded_state

    @staticmethod
    def decode(index):
        """Decodes an action index into a move (x, y, letter)."""
        x, y, letter_id = index // (BOARD_SIZE * 2), (index % (BOARD_SIZE * 2)) // 2, index % 2
        return x, y, 'S' if letter_id == 0 else 'O'

    def legal_moves(self):
        """Returns a list of all legal moves in the current board state."""
        return [(x, y, letter) for x in range(BOARD_SIZE) for y in range(BOARD_SIZE) if self.board[x, y] == ' ' for letter in ('S', 'O')]

    def status(self):
        """Returns the current game status: ongoing, winner, or draw."""
        if not self.game_over:
            return "ongoing"
        if self.scores[1] > self.scores[2]:
            return "Player 1 wins"
        elif self.scores[2] > self.scores[1]:
            return "Player 2 wins"
        return "Draw"

    def check_game_over(self):
        """Checks if the game is over by verifying if the board is full."""
        return np.all(self.board != ' ')

    def check_sos(self, x, y):
        seen_sos = set()  # מניעת כפל חישוב
        sos_coordinates = []  # רשימה של קואורדינטות של כל רצפי SOS שנמצאו
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # אנכי, אופקי, אלכסונים

        for dx, dy in directions:
            for start in [-2, -1, 0]:  # שלוש אפשרויות להתחלת רצף
                indices = tuple((x + (start + i) * dx, y + (start + i) * dy) for i in range(3))
                if all(self.is_valid(px, py) for px, py in indices):
                    if [self.board[px][py] for px, py in indices] == ['S', 'O', 'S']:
                        if indices not in seen_sos:  # מונע כפילויות
                            seen_sos.add(indices)
                            sos_coordinates.append(indices)  # שומרים את הקואורדינטות

        #print(f"Total SOS sequences found for ({x},{y}): {len(seen_sos)}")  # ⚡ הדפסה לבדיקת תקינות
        return len(seen_sos), sos_coordinates  # מחזירים את מספר הרצפים והרשימה שלהם

    def get_affected_positions(self, move):
        """ מחזירה רשימה של כל התאים שיכולים להיות מושפעים מהמהלך שנעשה ב-(x, y).
            מכסה גם תאים במרחק 1 וגם תאים במרחק 2 בכל כיוון. """
        x, y, letter = move

        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1), (0, 1),
                      (1, -1), (1, 0), (1, 1)]  # 8 הכיוונים האפשריים

        return [(x + dx * step, y + dy * step)  # חישוב נקודות מושפעות
                for dx, dy in directions for step in [1, 2]
                if self.is_valid(x + dx * step, y + dy * step) and
                self.board[x + dx * step, y + dy * step] == ' ']


    def display_board(self):
        """Displays the game board in a readable format."""
        print("  " + " ".join(map(str, range(BOARD_SIZE))))
        for i, row in enumerate(self.board):
            print(i, " ".join(row))

    @staticmethod
    def is_valid(x: int, y: int) -> bool:
        """Checks if the given coordinates are within the board's boundaries."""
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

    def computer_move(self):
        """Performs a random legal move for the computer."""
        legal_moves = self.legal_moves()
        if legal_moves:
            self.make_move(*random.choice(legal_moves))