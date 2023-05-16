import numpy as np


def apply_pessimism_to_transition(T, rewards_dict, scaling) -> np.ndarray:
    # Change the transition probabilities to be more pessimistic
    neg_rew_idx = [idx for idx in rewards_dict if rewards_dict[idx] < 0]

    T_new = T.copy()

    T_new[:, :, neg_rew_idx] *= scaling
    T_new /= T_new.sum(axis=2, keepdims=True)

    return T_new


def scale_from_pessimism(p_u, p=0.8):
    return (p_u * (1 - p)) / ((1 - p_u) * p)


if __name__ == "__main__":
    print(np.log2(scale_from_pessimism(0.4)))
    print(np.log2(scale_from_pessimism(0.99)))
