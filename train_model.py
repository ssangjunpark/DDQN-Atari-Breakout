import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import ale_py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


from util import ReplayMemory, ImageTransformer, smooth
from DDQN import DDQN, play_one

gym.register_envs(ale_py)

from constants import ACTION_SPACE_SIZE,  MAX_EXPERIENCE_BUFEER_SIZE, MIN_EXPERIENCE_BUFFER_SIZE, TARGET_UPDATE_PERIOD_CAP, TARGET_UPDATE_PERIOD_NEW, TARGET_UPDATE_PERIOD_OLD, IM_SIZE

def train_model():
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')

    conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1), (128, 3, 1)]
    dense_layer_sizes = [512, 256]
    gamma = 0.99
    batch_size = 32
    num_episodes = 3400
    total_step = 0 
    experience_replay_buffer = ReplayMemory(MAX_EXPERIENCE_BUFEER_SIZE, IM_SIZE, IM_SIZE, 4, 32)
    episode_rewards = np.zeros(num_episodes)


    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min) / 1000000

    image_transformer = ImageTransformer(IM_SIZE)

    model = DDQN(ACTION_SPACE_SIZE, conv_layer_sizes, dense_layer_sizes)
    target_model = DDQN(ACTION_SPACE_SIZE, conv_layer_sizes, dense_layer_sizes)

    # initilizing both models and copying parameters from model to target model
    model_init_zero_state = np.zeros((1, IM_SIZE, IM_SIZE, 4), dtype=np.float32)
    model.forward(model_init_zero_state)
    target_model.forward(model_init_zero_state)
    target_model.set_weights(model.get_weights())


    # initate and fill replay buffer to MIN_EXPERIENCES
    obs, _ = env.reset()
    
    for i in range(MIN_EXPERIENCE_BUFFER_SIZE):
        action = np.random.choice(ACTION_SPACE_SIZE)
        obs, reward, done, _, _ = env.step(action)
        obs_transformed = image_transformer.transform(obs)
        experience_replay_buffer.add_experience(action, obs_transformed, reward, done)

        if done:
            obs, _ = env.reset()
        
        if i % 100 == 0:
            print(f"{i}/{MIN_EXPERIENCE_BUFFER_SIZE} collected")

    # we will be recording video every 100 epochs
    env = RecordVideo(env, video_folder='ddqn_tf_atari', episode_trigger=lambda x : x % 100 == 0)
    
    target_update_period = TARGET_UPDATE_PERIOD_OLD
    
    t0 = datetime.now()
    for i in range(num_episodes):
        total_step, episode_reward, episode_duration, num_steps, epsilon = play_one(env, model, target_model, total_step, experience_replay_buffer,  image_transformer, gamma, batch_size, epsilon, epsilon_change, epsilon_min, target_update_period, ACTION_SPACE_SIZE)
        episode_rewards[i] = episode_reward

        running_average = episode_rewards[max(0, i-100):i + 1].mean()

        #slow down copy rate
        if total_step > TARGET_UPDATE_PERIOD_CAP:
            target_update_period = TARGET_UPDATE_PERIOD_NEW
        if i % 100 == 0:
            model.save_weights_custom(f'ddqn_weights{i}.h5')

        print(f"Episode: {i}, Episode Duration: {episode_duration}, Num Steps: {num_steps}, Reward: {episode_reward},  Running Average (100 prior episode max): {running_average:.3f}, Epsilon: {epsilon:.3f}")

    print(f"Total Duration: {datetime.now() - t0}")

    plt.plot(episode_rewards, label='episode rewards')
    plt.plot(smooth(episode_rewards), label='episode rewards smooth')
    plt.legend()
    plt.savefig('results.png')

    model.save_weights_custom('ddqn_weights.h5')

    env.close()