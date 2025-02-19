import tensorflow as tf
import numpy as np

class ImageTransformer:
    def __init__(self, IM_SIZE):
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
    def __init__(self, size, frame_height, frame_width, agent_history_length, batch_size):
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
    
def smooth(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - 99)
        y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
    return y

def epsilon_greedy(model, state, action_space_size, eps):
    if np.random.random() < eps:
        action = np.random.choice(action_space_size)
    else:
        action = np.argmax(model.forward(state))
    
    return action