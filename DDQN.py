import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import ale_py
from datetime import datetime
from util import epsilon_greedy
    
class DDQN:
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
    

def play_one(env, model, target_model, total_step, experiance_replay_buffer, image_transformer, gamma, batch_size, epsilon, epsilon_change, epsilon_min, target_update_period, action_space_size):
    start_time = datetime.now()
    obs, _ = env.reset()
    
    obs_transformed = image_transformer.transform(obs)
    # we stack 4 inital frame as our state // as shape (H x W x 4)
    state = np.stack([obs_transformed] * 4, axis=2)

    num_steps = 0
    episode_reward = 0

    done = False

    while not done:
        # copy the model parameter to target_model paramter iff
        if total_step % target_update_period == 0:
            target_model.set_weights(model.get_weights())
            print(f'Model weight copied to Target Model. Total Steps: {total_step}')

        
        # select action using epsilon greedy
        pi_state = np.expand_dims(state, axis=0).astype(np.float32)
        action = epsilon_greedy(model, pi_state, action_space_size, epsilon)


        obs, reward, done, _, _ = env.step(action)
        obs_transformed = image_transformer.transform(obs)
        next_state = np.append(state[:, :, 1:], np.expand_dims(obs_transformed, axis=2), axis=2)

        episode_reward += reward

        experiance_replay_buffer.add_experience(action, obs_transformed, reward, done)
        
        loss = model.learn(target_model, experiance_replay_buffer, gamma)
        
        num_steps += 1
        
        state = next_state
        total_step += 1

        epsilon = max(epsilon - epsilon_change, epsilon_min)
    
    return (total_step, episode_reward, (datetime.now()-start_time), num_steps, epsilon)