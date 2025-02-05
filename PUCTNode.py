class PUCTNode:
    
    #each node saves copy of the game state 
    def __init__(self, game, prior=1.0):
        self.game = game.clone()
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0
        self.children = {}

    def select(self, c_puct):
        best_score = -float('inf')
        best_action = None
        for action, child in self.children.items():
            u_score = child.total_value / (1 + child.visit_count) + c_puct * child.prior * (self.visit_count ** 0.5 / (1 + child.visit_count))
            if u_score > best_score:
                best_score = u_score
                best_action = action
        return best_action, self.children[best_action]

    def expand(self, action_probs):
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = PUCTNode(self.game, prob)

    def update(self, value):
        self.visit_count += 1
        self.total_value += value
        
        #TODO- asking Zur
        # 1. Use normalization for values: currently total_value is aggregated, 
        # it may be more useful to normalize it (self.total_value / self.visit_count) to avoid biases.
        # 2. Adding the Temperature mechanism: In Alpha-Zero, "Temperature" is used to control 
        # the level of exploration in different stages of the game.