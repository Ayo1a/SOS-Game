import random
import numpy as np
from constant import *
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
        x, y, letter, previous_player, points_scored = self.move_history.pop()
        self.board[x, y] = ' '
        self.scores[previous_player] -= points_scored
        self.current_player = previous_player
        self.game_over = False

    def clone(self):
        """Creates a deep copy of the current game state."""
        cloned_game = SOSGame()
        cloned_game.board = self.board.copy()
        cloned_game.current_player = self.current_player
        cloned_game.scores = self.scores.copy()
        cloned_game.game_over = self.game_over
        cloned_game.move_history = self.move_history.copy()
        return cloned_game

    def encode(self):
        """Encodes the game state as a one-hot tensor and extra information."""
        encoded_board = np.stack([
            np.array(self.board == 'S', dtype=np.float32),
            np.array(self.board == 'O', dtype=np.float32),
            np.array(self.board == ' ', dtype=np.float32)
        ])
        extra_info = np.array([self.current_player - 1, self.scores[1], self.scores[2], self.game_over], dtype=np.float32)
        return encoded_board, extra_info

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


# Game Loop
def play_game():
    """ריצה של המשחק עם השחקן האנושי והמחשב (עכשיו עם PUCT)."""
    game = SOSGame()
    game.set(board= np.array([
        [' ', 'O', 'O', ' ', ' '],
        ['S', 'S', ' ', 'S', 'O'],
        ['S', ' ', 'O', 'O', ' '],
        ['O', 'S', 'O', 'S', ' '],
        [' ', ' ', ' ', 'O', 'O']
    ], dtype=str), current_player=PLAYER_2, scores={PLAYER_1: 2, PLAYER_2: 1})

    puct_player = PUCTPlayer(c_puct=5.0, simulations=20)

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
            game.make_move(x, y, letter)

        print(f"Current scores - Player 1: {game.scores[PLAYER_1]}, Player 2: {game.scores[PLAYER_2]}")

    game.display_board()
    print(game.status())

if __name__ == "__main__":
    play_game()

