from PUCTNode import PUCTNode


def evaluate(game):
    if game.status()[0] == "Game Over":
        scores = game.status()[1]
        return 1 if scores[game.current_player] > scores[3 - game.current_player] else -1
    return 0

#TODO- asking Zur
# Currently the function only returns 1 / -1, but you can add returning soft values ​​(such as 0.5 in cases where the lead is not absolute).
# It is possible to improve evaluate so that it uses a separate evaluation network (Value Network) instead of relying only on the end state.


class PUCTPlayer:
    def __init__(self, c_puct=1.0, simulations=100):
        self.c_puct = c_puct
        self.simulations = simulations

    def play(self, game):
        root = PUCTNode(game)
        for _ in range(self.simulations):
            self.simulate(root)
        best_action = max(root.children.items(), key=lambda child: child[1].visit_count)[0]
        return best_action

    def simulate(self, node):
        if node.game.game_over:
            return evaluate(node.game)
        if not node.children:
            action_probs = {move: 1 / len(node.game.legal_moves()) for move in node.game.legal_moves()}
            node.expand(action_probs)
            return evaluate(node.game)
        action, child = node.select(self.c_puct)
        value = self.simulate(child)
        child.update(value)
        return -value
    
    Disadvantages and possible improvements:

#TODO- asking Zur
# Evaluate improvement:
# Currently the evaluation is only suitable for finite states.
# It is worth integrating a Value Network that also provides a value for non-finite states.
# Replace the uniform probabilities (1 / len(moves)) with a real Policy Network
# Currently, the expansion of nodes is made in general (as if a move has the same probability).
# Solution: Use a Neural Network that will provide smarter probabilities.
# Cleaning old nodes in the tree:
# Currently, there is no mechanism that deletes old nodes → memory grows too quickly.
# Solution: Reset the Tree every few steps to prevent excess memory usage.
# Parallel game (parallel MCTS) can be integrated to speed up the search.
