class SOSGame:
    PLAYER_1 = 1
    PLAYER_2 = 2

    def __init__(self):
        self.board = [[' ' for _ in range(8)] for _ in range(8)]
        self.current_player = SOSGame.PLAYER_1
        self.scores = {SOSGame.PLAYER_1: 0, SOSGame.PLAYER_2: 0}
        self.game_over = False

    def make_move(self, x, y, letter):
        if self.board[x][y] != ' ' or letter not in ('S', 'O'):
            raise ValueError("Invalid move")
        self.board[x][y] = letter
        points = self.check_sos(x, y)
        self.scores[self.current_player] += points
        if points == 0:
            self.current_player = SOSGame.PLAYER_1 if self.current_player == SOSGame.PLAYER_2 else SOSGame.PLAYER_2
        self.game_over = self.check_game_over()

    def unmake_move(self, x, y):
        self.board[x][y] = ' '
        self.game_over = False

    def clone(self):
        clone = SOSGame()
        clone.board = [row[:] for row in self.board]
        clone.current_player = self.current_player
        clone.scores = self.scores.copy()
        clone.game_over = self.game_over
        return clone

    def encode(self):
        #TODO: changing 
        flat_board_s = [1 if cell == 'S' else 0 for row in self.board for cell in row]
        flat_board_o = [1 if cell == 'O' else 0 for row in self.board for cell in row]
        # flat_board = [1 if cell == 'S' else 2 if cell == 'O' else 0 for row in self.board for cell in row]
        return flat_board_s + flat_board_o + [self.current_player, self.scores[SOSGame.PLAYER_1], self.scores[SOSGame.PLAYER_2]]

    @staticmethod
    def decode(action_index):
        #TODO: changing 
        x = action_index // 16
        y = (action_index % 16) // 2
        letter = 'S' if action_index % 2 == 0 else 'O'
        return x, y, letter

    def legal_moves(self):
        return [(x, y, letter) for x in range(8) for y in range(8) if self.board[x][y] == ' ' for letter in ('S', 'O')]

    def status(self):
        if self.game_over:
            return "Game Over", self.scores
        return "Ongoing"

    def check_sos(self, x, y):
        points = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            for start in [-2, -1, 0]:
                seq = [self.board[x + (start + i) * dx][y + (start + i) * dy] \
                       for i in range(3) \
                       if 0 <= x + (start + i) * dx < 8 and 0 <= y + (start + i) * dy < 8]
                if seq == ['S', 'O', 'S']:
                    points += 1
        return points

    def check_game_over(self):
        return all(self.board[x][y] != ' ' for x in range(8) for y in range(8))

''' checked -  we don't really need it '''
#TODO:
''' unmake_move(x, y) does not restore the score '''
# What is the problem?
# In the current version, the function unmake_move(x, y) only deletes the letter from the board, but does not restore the score to its original state.
# If we make a move where we created an "SOS" and received points, undoing the move will leave the score as if the move still existed!

# Suggested improvement
# We will add a data structure that will store the changes in the score, so that we can restore them when undoing a move.
# We will add a stack for each player to track the changes in the score.

''' check_game_over() could be smarter '''
# What's the problem?
# Currently check_game_over() only ends the game when the entire board is full, but the game may be lost for both players before then.

# Suggested improvement
# We will check if there are any moves left that could create an "SOS", not just if the board is full.

''' check_sos(x, y) could be more efficient '''
# What's the problem?
# Currently the function check_sos(x, y) rescans all directions on each move, even if certain options are known to be irrelevant.
# This could be improved by pre-marking squares that could create an "SOS", and only scanning the area around the last move.
# Suggested improvement
# Instead of checking the entire board each time, we will only check the area closest to the last move.