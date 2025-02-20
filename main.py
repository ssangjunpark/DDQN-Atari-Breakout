from train_model import train_model
from test_model import test_model


if __name__ == "__main__":
    train_model(conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1), (128, 3, 1)], dense_layer_sizes = [512, 256], gamma = 0.99, batch_size = 32, num_episodes = 3400, epsilon = 1.0, epsilon_min = 0.1)
    # test_model('dqn_weights3300.h5', 100, conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1), (128, 3, 1)], dense_layer_sizes=[512, 256])