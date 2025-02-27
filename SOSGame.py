import numpy as np

class SOSGame:
    PLAYER_1 = 1
    PLAYER_2 = 2

    def __init__(self):
        """Initializes the game board and player settings."""
        self.board = np.full((8, 8), ' ', dtype=str)
        self.current_player = SOSGame.PLAYER_1
        self.scores = {SOSGame.PLAYER_1: 0, SOSGame.PLAYER_2: 0}
        self.game_over = False
        self.move_history = []

    def make_move(self, x, y, letter):
        """Executes a move and updates the score. The turn always switches after a move."""
        if self.board[x, y] != ' ' or letter not in ('S', 'O'):
            raise ValueError("Invalid move")

        self.board[x, y] = letter
        points_scored = self.check_sos(x, y)
        self.scores[self.current_player] += points_scored
        self.move_history.append((x, y, letter, self.current_player, points_scored))

        if self.check_game_over():
            self.game_over = True
        else:
            self.current_player = SOSGame.PLAYER_1 if self.current_player == SOSGame.PLAYER_2 else SOSGame.PLAYER_2

    def unmake_move(self):
        """Reverts the board to the previous state by undoing the last move."""
        if not self.move_history:
            return
        x, y, letter, previous_player, points_scored = self.move_history.pop()
        self.board[x, y] = ' '
        self.scores[previous_player] -= points_scored
        self.current_player = previous_player
        self.game_over = False

    def encode(self):
        """Encodes the game state as a one-hot tensor and extra information."""
        s_map = np.array(self.board == 'S', dtype=np.float32)
        o_map = np.array(self.board == 'O', dtype=np.float32)
        empty_map = np.array(self.board == ' ', dtype=np.float32)
        extra_info = np.array([self.current_player - 1, self.scores[1], self.scores[2], self.game_over], dtype=np.float32)
        return np.stack([s_map, o_map, empty_map]), extra_info

    @staticmethod
    def decode(index):
        """Decodes an action index into a move (x, y, letter)."""
        x, y, letter_id = index // 16, (index % 16) // 2, index % 2
        return x, y, 'S' if letter_id == 0 else 'O'

    def legal_moves(self):
        """Returns a list of all legal moves in the current board state."""
        return [(x, y, letter) for x in range(8) for y in range(8) if self.board[x, y] == ' ' for letter in ('S', 'O')]

    def status(self):
        """Returns the current game status: ongoing, winner, or draw."""
        if not self.game_over:
            return "ongoing"
        if self.scores[1] > self.scores[2]:
            return "Player 1 wins"
        elif self.scores[2] > self.scores[1]:
            return "Player 2 wins"
        else:
            return "Draw"

    def check_game_over(self):
        """Checks if the game is over by verifying if the board is full."""
        return np.all(self.board != ' ')

    def check_sos(self, x, y):
        """Checks if a move creates an SOS sequence and returns the points scored."""
        points = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Horizontal, vertical, diagonal directions

        for dx, dy in directions:
            # Case 1: The new letter is 'O' at the center (S O S)
            if self.is_valid(x - dx, y - dy) and self.is_valid(x + dx, y + dy):
                if self.board[x - dx, y - dy] == 'S' and self.board[x, y] == 'O' and self.board[x + dx, y + dy] == 'S':
                    points += 1

            # Case 2: The new letter is 'S' at the end of (S O S)
            if self.is_valid(x - 2 * dx, y - 2 * dy) and self.is_valid(x - dx, y - dy):
                if self.board[x - 2 * dx, y - 2 * dy] == 'S' and self.board[x - dx, y - dy] == 'O' and self.board[x, y] == 'S':
                    points += 1

        return points

    @staticmethod
    def is_valid(x, y):
        """Checks if the given coordinates are within the board's boundaries."""
        return 0 <= x < 8 and 0 <= y < 8
