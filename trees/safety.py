from trees.models import Node, Leaf, State


CONSUMPTION = [
	[ 9.615087, 5.795526, 4.603187, 4.542417, 6.138689, 16.391365, 43.207597, 42.00412, 34.003248, 31.988944, 28.477089, 25.785929, 24.628457, 23.417857, 22.774494, 25.163429, 29.21139, 34.671253, 39.552415, 37.152861, 29.291631, 26.221253, 23.878959, 15.738388 ],
	[ 9.689989, 5.794858, 4.709297, 4.47221, 6.012476, 16.23697, 42.529266, 40.818582, 32.369595, 30.124602, 26.91396, 24.820371, 23.523606, 22.197558, 22.525408, 24.829038, 29.539078, 34.509104, 38.142511, 36.981069, 30.480625, 26.689531, 23.551861, 16.271192 ],
	[ 9.834889, 5.876961, 4.53861, 4.430997, 5.956947, 16.386985, 43.32831, 41.422095, 33.066159, 31.159582, 28.408058, 25.820132, 24.722594, 23.152701, 23.034695, 24.666871, 29.792432, 34.99171, 38.932862, 37.619877, 30.923747, 26.902407, 23.317901, 15.834372 ],
	[ 9.586711, 5.707676, 4.53154, 4.484971, 6.355857, 16.490495, 42.922208, 41.324143, 33.811094, 31.83512, 28.683098, 26.061707, 24.034984, 22.364339, 22.356159, 24.641306, 29.224471, 34.144769, 37.123919, 36.253432, 29.852602, 26.844322, 23.912588, 16.031639 ],
	[ 9.870218, 5.819499, 4.701071, 4.590401, 6.124585, 16.155316, 41.513834, 40.929327, 34.354539, 32.870113, 29.938488, 27.59474, 26.101001, 24.776526, 24.920676, 27.181678, 30.700727, 32.928773, 33.523951, 30.409019, 24.436347, 22.972663, 20.808673, 17.989527 ],
	[ 12.13667, 7.377348, 5.235036, 4.5965, 4.645113, 6.406783, 12.706801, 26.931674, 42.352011, 51.380389, 50.634058, 46.826956, 43.02649, 38.621774, 34.652988, 32.680335, 33.089229, 33.656452, 32.92201, 30.701944, 25.687736, 22.19035, 20.489378, 18.00798 ],
	[ 13.026306, 7.924664, 5.571639, 4.812678, 4.384187, 5.908583, 10.942723, 23.663847, 39.559533, 49.590895, 50.23242, 47.996505, 44.537656, 39.96267, 35.908773, 33.970254, 34.658137, 37.807594, 39.872124, 38.736026, 31.432597, 27.258044, 23.159028, 16.463232 ]
]

ACTIONS = {
    '1': 0,
    '2': 50,
    '3': 55,
    '4': 60,
    '5': 65,
    '6': 70,
    '7': 75,
    '8': 80,
    '9': 85,
    '10': 90,
    '11': 95,
    '12': 100,
    '13': 105,
    '14': 110,
    '15': 115,
    '16': 120
}

class EnvState:
    def __init__(self, init_tank=0, init_action='1', init_hour=0):
        self.tank = init_tank
        self.action = init_action
        self.hour = init_hour
        self.minute = 0
        self.consumpion = CONSUMPTION[init_hour] + sel


def get_initial_concrete_state(init_tank=0, init_action='1', init_hour=0):
    """
    State is [T, F, D, hour]
    """
    return {
        'T': init_tank,
        'F': ACTIONS[init_action],
        'D': CONSUMPTION[init_hour],
        'chour': init_hour
    }


def next_symbolic_state(state, action):
    """
    Takes a concrete state and an action to perform, and returns the symbolic
    state that results from taking this action. The symbolic state is given as
    a list of tuples that defines the intervals of possible values of each
    variable in the state.
    """
    pass


def get_possible_actions(tree, state):
    """
    Takes a strategy represented as a tree aswell as a symbolic state and
    returns all the possible actions that can be chosen by the strategy for
    this given symbolic state.
    """
    # the state that is returned does not really matter, does it?

    return tree.get_leafs_at_symbolic_state(state, pairs=[])

def main(tree):
    state = get_initial_state()
    action = tree.get_leaf(state)
    sym_state = next_symbolic_state(state, action)
    as_pairs = get_possible_actions(tree, sym_state)

    # need to do something with keeping track of action paths
