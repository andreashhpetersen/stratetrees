import numpy as np


class ShieldedTree:
    def __init__(self, tree, shield, action_map):
        self.tree = tree
        self.shield = shield
        self.action_map = action_map
        self.n_actions = len(tree.actions)
        self.corrections = 0

    def predict(self, obs):
        allowed_acts = self.action_map[self.shield.predict(obs)]
        optimal_act = self.tree.predict(obs)

        # return optimal action if its safe or if none are safe
        if optimal_act in allowed_acts or len(allowed_acts) == 0:
            return optimal_act

        # otherwise, choose randomly between the allowed actions
        else:
            self.corrections += 1
            return allowed_acts[np.random.randint(len(allowed_acts))]
