import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import ale_py
from datetime import datetime

gym.register_envs(ale_py)


MAX_EXPERIENCES = 500000
MIN_EXPERIENCES = 50000
TARGET_UPDATE_PERIOD_CAP = 100000
IM_SIZE = 84
ACTION_SPACE = 4 # env.action_space.n but we are not considering the last 2 since it does not play any impact on playing game

class ImageTransformer:
    def __init__(self):
        self.im_size = IM_SIZE

    def transform(self, state):
        # we want to convert image to grayscale final output should be of size (H x W) we will be droping the color channel since we will be stacking it 
        gray_scaled = tf.image.rgb_to_grayscale(state) # the last dimension C is 1

        cropped = tf.image.crop_to_bounding_box(gray_scaled, 34, 0, 160, 160)
        
        resized = tf.image.resize(cropped, [self.im_size, self.im_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # we will be getting rid of the C dimension
        return tf.squeeze(resized).numpy().astype(np.float32)
    
# Replay memory inspired form https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
class ReplayMemory:
    def __init__(self, size=MAX_EXPERIENCES, frame_height=IM_SIZE, frame_width=IM_SIZE, agent_history_length=4, batch_size=32):
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        # pointer to locate where we are in the memory
        self.current = 0

        # place holder for buffer / pre allocating memory to prevent leak during training
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.float32)
        self.terminal_flags = np.empty(self.size, dtype=np.bool_)

        # to store the batch we sample
        self.batch_states = np.empty((self.batch_size, self.agent_history_length, self.frame_height, self.frame_width), dtype=np.float32)
        self.batch_new_states = np.empty((self.batch_size, self.agent_history_length, self.frame_height, self.frame_width), dtype=np.float32)
        self.batch_indices = np.empty(self.batch_size, dtype=np.int32)

    # a, s', r, is s' terminal // we dont explicitely have to include s since replay memory can be used to identify s
    def add_experience(self, action, frame, reward, terminal):
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is incorrect')
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        # use self.current as a pointer 
        self.frames[self.current, ...] = frame
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        if self.count < self.agent_history_length:
            raise ValueError('Replay Buffer is too small!')
        if index < self.agent_history_length - 1:
            raise ValueError('Index must be greater or equal to agent_history_length - 1')
        return self.frames[index - self.agent_history_length + 1 : index + 1, ...]
    
    # we want to ensure that the selected pointer/indices to ensure the selected batch of 4 consecutive frames have correct time stamp
    def _get_valid_indicides(self):
        for i in range(self.batch_size):
            while True:
                sampled_idx = np.random.randint(self.agent_history_length-1, self.count)
                if sampled_idx < self.agent_history_length:
                    continue
                if sampled_idx >= self.current and sampled_idx - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[sampled_idx - self.agent_history_length:sampled_idx].any():
                    continue
                break
            self.batch_indices[i] = sampled_idx

    def get_minibatch(self):
        self._get_valid_indicides()

        # state = [t-4:t] state' = [t-3:t+1]
        for i, idx in enumerate(self.batch_indices):
            self.batch_states[i] = self._get_state(idx - 1)
            self.batch_new_states[i] = self._get_state(idx)

        # transposed to match tensorflows N x H x W x C , where C = 4 \ original was N x C x H x W
        batch_states = np.transpose(self.batch_states, axes=(0, 2, 3, 1)).astype(np.float32)
        batch_next_states = np.transpose(self.batch_new_states, axes=(0, 2, 3, 1)).astype(np.float32)

        return (batch_states, self.actions[self.batch_indices], self.rewards[self.batch_indices], batch_next_states, self.terminal_flags[self.batch_indices])

class DQN:
    def __init__(self, action_space_size, conv_layer_sizes, dense_layer_sizes):
        self.action_space_size = action_space_size
        self.conv_layer_sizes = conv_layer_sizes
        self.dense_layer_sizes = dense_layer_sizes
        

        # creating model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=2.5e-4, epsilon=1e-8, amsgrad=True)
        self.model = tf.keras.Sequential()

        # add convoltiion layers
        for filters, kernal_size, strides in self.conv_layer_sizes:
            self.model.add(
                tf.keras.layers.Conv2D(filters, kernal_size, strides, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(1e-6))
            )

        # flatten the conv layer data to be N x (H * w * C)instefad of N x H x W x Cs
        self.model.add(tf.keras.layers.Flatten())
        
        # add the dense layers
        for units in self.dense_layer_sizes:
            self.model.add(
                tf.keras.layers.Dense(units, activation='relu', kernel_initializer='he_uniform')
            )
            

        # append the final layer for approximation of Q(s,a) for all a in A
        self.model.add(tf.keras.layers.Dense(units=self.action_space_size, kernel_initializer='he_uniform'))


    def forward(self, states):
        states = states / 255.0 # normalize yey
            
        return self.model(states)
    
    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def save_weights_custom(self, name):
        self.model.save_weights(name)

    def load_weights_custom(self, name):
        self.model.load_weights(name)
        
    @tf.function
    def train_step(self, states, actions, targets):
        with tf.GradientTape() as tape:
            current_q = self.forward(states)
            one_hot_mask = tf.one_hot(actions, self.action_space_size) # since we are only interested in Q(s,a) where a is the selection action in state s
            q_with_selected_action = tf.reduce_sum(current_q * one_hot_mask, axis=1)
            loss = tf.reduce_mean(tf.losses.huber(targets, q_with_selected_action))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    #Double DQN
    def learn(self, target_model, experience_replay_buffer, gamma):
        states, actions, rewards, next_states, dones = experience_replay_buffer.get_minibatch()

        next_q_online = self.forward(next_states)
        next_actions = tf.argmax(next_q_online, axis=1)

        next_q_target = target_model.forward(next_states)
        next_q_values = tf.gather(next_q_target, next_actions, batch_dims=1)

        targets = rewards + tf.cast(tf.logical_not(dones), tf.float32) * gamma * next_q_values

        loss = self.train_step(states, actions, targets)
        return loss
    

def epsilon_greedy(model, state, eps):
    if np.random.random() < eps:
        action = np.random.choice(ACTION_SPACE)
    else:
        action = np.argmax(model.forward(state))
    
    return action

def play_one(env, total_t, experiance_replay_buffer, model, target_model, image_transformer, gamma, batch_size, epsilon, epsilon_change, epsilon_min, target_update_period):
    t0 = datetime.now()
    obs, _ = env.reset()
    
    obs_transformed = image_transformer.transform(obs)
    # we stack 4 inital frame as our state // as shape (H x W x 4)
    state = np.stack([obs_transformed] * 4, axis=2)

    total_time_taken = 0
    num_steps = 0
    episode_reward = 0

    done = False

    while not done:
        # copy the model parameter to target_model paramter iff
        if total_t % target_update_period == 0:
            target_model.set_weights(model.get_weights())
            print(f'Model weight copied to Target Model. Total Time: {total_t}')

        
        # select action using epsilon greedy
        pi_state = np.expand_dims(state, axis=0).astype(np.float32)
        action = epsilon_greedy(model, pi_state, epsilon)


        obs, reward, done, _, _ = env.step(action)
        obs_transformed = image_transformer.transform(obs)
        next_state = np.append(state[:, :, 1:], np.expand_dims(obs_transformed, axis=2), axis=2)

        episode_reward += reward

        experiance_replay_buffer.add_experience(action, obs_transformed, reward, done)


        t1 = datetime.now()
        
        loss = model.learn(target_model, experiance_replay_buffer, gamma)
        
        dt = datetime.now() - t1
        total_time_taken += dt.total_seconds()

        num_steps += 1
        
        state = next_state
        total_t += 1

        epsilon = max(epsilon - epsilon_change, epsilon_min)
    
    return (total_t, episode_reward, (datetime.now()-t0), num_steps, total_time_taken/num_steps, epsilon)


def smooth(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - 99)
        y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
    return y

def main():
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')

    conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1), (128, 3, 1)]
    dense_layer_sizes = [512, 256]
    gamma = 0.99
    batch_size = 32
    num_episodes = 3400
    total_t = 0 
    experience_replay_buffer = ReplayMemory()
    episode_rewards = np.zeros(num_episodes)


    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min) / 1000000

    image_transformer = ImageTransformer()

    model = DQN(ACTION_SPACE, conv_layer_sizes, dense_layer_sizes)
    target_model = DQN(ACTION_SPACE, conv_layer_sizes, dense_layer_sizes)

    # initilizing both models and copying parameters from model to target model
    model_init_zero_state = np.zeros((1, IM_SIZE, IM_SIZE, 4), dtype=np.float32)
    model.forward(model_init_zero_state)
    target_model.forward(model_init_zero_state)
    target_model.set_weights(model.get_weights())


    # initate and fill replay buffer to MIN_EXPERIENCES
    obs, _ = env.reset()
    
    for i in range(MIN_EXPERIENCES):
        action = np.random.choice(ACTION_SPACE)
        obs, reward, done, _, _ = env.step(action)
        obs_transformed = image_transformer.transform(obs)
        experience_replay_buffer.add_experience(action, obs_transformed, reward, done)

        if done:
            obs, _ = env.reset()
        
        if i % 100 == 0:
            print(f"{i}/{MIN_EXPERIENCES} collected")

    # we will be recording video every 100 epochs
    env = RecordVideo(env, video_folder='dqn_tf_atari', episode_trigger=lambda x : x % 100 == 0)
    
    target_update_period = 5000
    
    t0 = datetime.now()
    for i in range(num_episodes):
        total_t, episode_reward, episode_duration, num_steps, average_time_per_step, epsilon = play_one(env, total_t, experience_replay_buffer, model, target_model, image_transformer, gamma, batch_size, epsilon, epsilon_change, epsilon_min, target_update_period)
        episode_rewards[i] = episode_reward

        running_average = episode_rewards[max(0, i-100):i + 1].mean()

        #slow down copy rate
        if total_t > TARGET_UPDATE_PERIOD_CAP:
            target_update_period = 10000
        if i % 100 == 0:
            model.save_weights_custom(f'dqn_weights{i}.h5')

        print(f"Episode: {i}, Episode Duration: {episode_duration}, Num Steps: {num_steps}, Reward: {episode_reward}, Average Time Per Step: {average_time_per_step:.3f}, Running Average (100 prior episode max): {running_average:.3f}, Epsilon: {epsilon:.3f}")

    print(f"Total Duration: {datetime.now() - t0}")

    plt.plot(episode_rewards, label='episode rewards')
    plt.plot(smooth(episode_rewards), label='episode rewards smooth')
    plt.legend()
    plt.savefig('results.png')

    model.save_weights_custom('dqn_weights.h5')

    env.close()

def test_model(name, sample_quantity):
    max_reward = 0
    max_reward_idx = 0
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    env = RecordVideo(env, video_folder='dqn_tf_atari_improved_test_model', episode_trigger=lambda x : True)
    image_transformer = ImageTransformer()

    model = DQN(4, [(32, 8, 4), (64, 4, 2), (64, 3, 1), (128, 3, 1)], [512, 256])
    
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
            action = epsilon_greedy(model, pi_state, 0.01)

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


if __name__ == "__main__":
    with tf.device('/gpu:0'):
        main()
        # test_model('dqn_weights3300.h5', 100)