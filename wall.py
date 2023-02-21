from mdp import *
import os


def wall(height, width, wall_width, wall_height, neg_mag, reward_mag, latent_cost=0):
    reward_dict = {}
    for i in range(height * width):
        reward_dict[i] = latent_cost # add latent cost
    wall_end_x = width - 1
    wall_begin_x = wall_end_x - wall_width
    wall_end_y = wall_height
    wall_begin_y = 0
    for i in range(wall_begin_x, wall_end_x):
        for j in range(wall_begin_y, wall_end_y):
            reward_dict[width*j + i] = neg_mag
    reward_dict[width - 1] = reward_mag
    return reward_dict


if __name__ == '__main__':

    default_prob = 0.8
    sns.set()

    setup_name = 'Wall'
    setup_name = setup_name.replace(' ', '_').lower()

    if not os.path.exists(f'images/{setup_name}'):
        os.makedirs(f'images/{setup_name}')

    height = 10
    width = 5


    wall_dict = wall(height, width, wall_width=3, wall_height=9, neg_mag=-10, reward_mag=100, latent_cost=-1)
    test = Experiment_2D(10, 5, rewards_dict=wall_dict)
    test.mdp.solve(setup_name=setup_name, policy_name='Baseline World')
    test.mdp.reset()

    # MYOPIC EXPERIMENT RUNS:
    for gamma in np.arange(0.5, 0.99, 0.1):
        test.mdp.reset()
        myopic = test.myopic(gamma=gamma)
        test.mdp.solve(setup_name=setup_name, policy_name='Myopic Agent: \u03B3={:.3f}'.format(gamma))

    # UNDERCONFIDENT + OVERCONFIDENT EXPERIMENT RUNS:
    for prob in np.arange(0.05, 0.5, 0.05):
        test.mdp.reset()
        confident = test.confident(make_right_prob=prob)
        if prob < default_prob:
            test.mdp.solve(setup_name=setup_name, policy_name='Underconfident Agent: p={:.3f}'.format(prob))
        elif prob > default_prob:
            test.mdp.solve(setup_name=setup_name, policy_name='Overconfident Agent: p={:.3f}'.format(prob))