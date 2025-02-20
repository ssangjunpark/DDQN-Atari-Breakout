import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt

from util import ImageTransformer, epsilon_greedy
from DDQN import DDQN

from constants import IM_SIZE, ACTION_SPACE_SIZE


def test_model(name, sample_quantity, conv_layer_sizes, dense_layer_sizes):
    max_reward = 0
    max_reward_idx = 0
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    env = RecordVideo(env, video_folder='ddqn_tf_atari_improved_test_model', episode_trigger=lambda x : True)
    image_transformer = ImageTransformer(IM_SIZE)

    model = DDQN(ACTION_SPACE_SIZE, conv_layer_sizes, dense_layer_sizes)
    
    # initialize model variables 
    model_init_zero_state = np.zeros((1, IM_SIZE, IM_SIZE, 4), dtype=np.float32)
    model.forward(model_init_zero_state)

    #load weights
    model.load_weights_custom(name)

    for i in range(sample_quantity):
        episode_reward = 0
        done = False

        obs, _ = env.reset()
        obs_transformed = image_transformer.transform(obs)
        state = np.stack([obs_transformed] * 4, axis=2)

        while not done:
            pi_state = np.expand_dims(state, axis=0).astype(np.float32)
            action = epsilon_greedy(model, pi_state, ACTION_SPACE_SIZE, 0.01)

            obs, reward, done, _, _ = env.step(action)
            obs_transformed = image_transformer.transform(obs)
            state = np.append(state[:, :, 1:], np.expand_dims(obs_transformed, axis=2), axis=2)

            episode_reward += reward

        print(f'Episode Reward: {episode_reward}')
        if episode_reward > max_reward:
            max_reward = episode_reward
            max_reward_idx = i
    
    print("Max Reward Video Index: ", max_reward_idx)
    env.close()

    return episode_reward