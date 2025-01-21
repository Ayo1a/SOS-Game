from PUCTNode import PUCTNode


def evaluate(game):
    if game.status()[0] == "Game Over":
        scores = game.status()[1]
        return 1 if scores[game.current_player] > scores[3 - game.current_player] else -1
    return 0


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
