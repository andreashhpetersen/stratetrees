import numpy as np

from numpy.typing import ArrayLike

from stratetrees.loaders import UppaalLoader
from stratetrees.nodes import Leaf


class QTree:
    def __init__(self, path: str):
        roots, actions, variables, meta = UppaalLoader.load(path)
        self.variables = variables
        self.meta = meta

        self.act2id = { a: i for i, a in enumerate(actions) }
        self.actions = actions
        self.roots = roots

        for r, a in zip(roots, actions):
            for leaf in r.get_leaves():
                leaf.action = self.act2id[a]

        self._size = sum([r.size for r in self.roots])

    @property
    def size(self):
        """Get size of the tree in the number of leaves"""
        if not hasattr(self, '_size') or self._size is None:
            self._size = sum([r.size for r in self.roots])
        return self._size

    def predict(self, state: ArrayLike, maximize=False) -> int:
        """
        Predict the best action based on a `state`.

        Parameters
        ----------
        state : array_like
            Input state
        maximize : bool, optional
            If set to True, return the action with the largest Q value

        Returns
        ------
        action_id : int
            The index of the preferred action
        """
        qs = self.predict_qs(state)
        return np.argmax(qs) if maximize else np.argmin(qs)

    def predict_qs(self, state: ArrayLike) -> np.ndarray:
        """
        Predict the Q values for each action given `state`.

        Parameters
        ----------
        state : array_like
            Input state

        Returns
        -------
        q_values : numpy.ndarry
            An array of Q values for each action
        """
        return np.array([r.get(state).cost for r in self.roots])

    def leaves(self, sort='min') -> list[Leaf]:
        """
        Get all the leaves of the roots of this Q tree.

        Parameters
        ----------
        sort : str
            If set to `'min'` (default), sort the leaves ascendingly according
            to cost. If set to `'max'` sort descendingly. Otherwise, no sorting
            is applied.

        Returns
        -------
        leaves : list
            The list of all leaves in the Q tree.
        """
        leaves = [l for ls in [r.get_leaves() for r in self.roots] for l in ls]
        if sort == 'min':
            leaves.sort(key=lambda x: x.cost)
        elif sort == 'max':
            leaves.sort(key=lambda x: -x.cost)
        return leaves
