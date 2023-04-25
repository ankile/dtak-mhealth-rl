import numpy as np


def apply_pessimism_to_transition(T, rewards_dict, scaling) -> np.ndarray:
    # Change the transition probabilities to be more pessimistic
    neg_rew_idx = [idx for idx in rewards_dict if rewards_dict[idx] < 0]

    T_new = T.copy()

    T_new[:, :, neg_rew_idx] *= scaling
    T_new /= T_new.sum(axis=2, keepdims=True)

    return T_new
