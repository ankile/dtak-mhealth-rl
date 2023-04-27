import numpy as np


def transition_matrix_is_valid(transition_matrix) -> bool:
    """
    Check if the transition matrix is valid.
    The transition matrix has shape (a, n, n), where a is the number of actions,
    n is the number of states.
    """
    if not isinstance(transition_matrix, np.ndarray):
        return False

    if transition_matrix.ndim != 3:
        return False

    if transition_matrix.shape[0] == 0:
        return False

    if transition_matrix.shape[1] == 0:
        return False

    if transition_matrix.shape[2] == 0:
        return False

    if not np.allclose(transition_matrix.sum(axis=2), 1):
        return False

    return True


def make_absorbing(T, idx):
    for a in range(4):
        for j in range(T.shape[2]):
            T[a, idx, j] = int(idx == j)


def id_func(T, *args, **kwargs):
    return T
