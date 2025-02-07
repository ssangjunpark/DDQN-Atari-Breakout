# DDQN-Atari-Breakout
TensorFlow 2 Implementation of Double Deep Q Network for Atari Breakout. 

**The original paper can be found [here](https://arxiv.org/pdf/1509.06461).**

This implementation utilizes Farama's [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environment, an open source Python library for Reinforment Learning agent training. Specifically, this implementation focuses on maximizing the expected future reward (namely the Q-Value Function) within the Emulator of Atari's ROM.

Q-Function is defined as the expected value of future rewards when taking action sampled from our policy $\pi$ (Epsilon Greedy with decaying epsilon will be used to balance exploration and exploitation).

The Q-value functon is defined as below in the original paper:

$$Q^\pi(s, a) \equiv \mathbb{E} \left[ R_1 + \gamma R_2 + \dots \mid S_0 = s, A_0 = a, \pi \right]$$


Our Q-value functon, which is approximated with the help of neural network, is defined over contious state space S and discrete action space A.

The gradient descent method will be used to minimize Temporal Differce error $\delta_t = Y^{DoubleQ}_t - Q(S_t, A_t; \theta_t)$:

$$\theta_{t+1} = \theta_t + \alpha\big(Y^{DoubleQ}t - Q(S_t, A_t; \theta_t)\big) \nabla{\theta_t} Q(S_t, A_t; \theta_t)$$

- $Y^{DoubleQ}$ should be subscripted with $t$

where the target $Y^{DoubleQ}_t$ is defined as:

$$Y^{\text{DoubleQ}}{t} \equiv R_{t+1} + \gamma Q(S_{t+1}, \underset{a}{\text{arg max}} , Q(S_{t+1}, a; \theta_t); \theta'_t)$$

- $Y^{DoubleQ}$ should be subscripted with $t$

Where $\theta_t$ is our online weights (from our main Q-Value estimate) and $\theta'_t$ is target Q-Value estimate weight which is copied from the main Q-Value Network every $\tau$ time-steps.

DDQM improves upon the DQN by using two seperate Q-value estimate (with weights $\theta_t$ and $\theta'_t$) to estmate Q-value for $Y^{DoubleQ}_t$.

## Aditional implementaion details
* Replay Memory Buffer
* Image Transformer
* Custom DDQN

## Sample Episode (After 3300 episodes of training)

[![Watch the video](https://img.youtube.com/vi/Bu5OjPkXfDc/0.jpg)](https://youtu.be/Bu5OjPkXfDc)
