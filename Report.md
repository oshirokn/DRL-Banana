# Project 1: Navigation - Report

# Introduction

In this project, I trained an agent to navigate and collect bananas in a large, square world.  

# Environment

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

# Agent Implementation

## Methods

The agent uses a [Deep Q-Learning Algorithm](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). 
It is a Value-Based method, combining Q-Learning reinforcement learning (SARSAMAX) and a Deep Neural Network to update the Q-Table.

*Bellman equation in Q-Learning*
![image](https://user-images.githubusercontent.com/24456678/135746096-89a65c81-daa7-4d05-a228-055a73e17c97.png)

This implementation also includes 2 improvements over the original paper: 
- Experience Replay

> When the agent interacts with the environment, the sequence of experience tuples can be highly correlated. 
> The naive Q-learning algorithm that learns from each of these experience tuples in sequential order runs the risk of getting swayed by the effects of this correlation. 
> By instead keeping track of a replay buffer and using experience replay to sample from the buffer at random, we can prevent action values from oscillating or diverging catastrophically.
> The replay buffer contains a collection of experience tuples (SS, AA, RR, S'S′). The tuples are gradually added to the buffer as we are interacting with the environment.
> The act of sampling a small batch of tuples from the replay buffer in order to learn is known as experience replay. In addition to breaking harmful correlations, experience replay allows us to learn more from individual tuples multiple times, recall rare occurrences, and in general make better use of our experience.

- Fixed Q Targets
> In Q-Learning, we update a guess with a guess, and this can potentially lead to harmful correlations. To avoid this, we can update the parameters ww in the network \hat{q}q^​ to better approximate the action value corresponding to state SS and action AA with the following update rule:
> here w^-w − are the weights of a separate target network that are not changed during the learning step, and (SS, AA, RR, S'S′) is an experience tuple.

![image](https://user-images.githubusercontent.com/24456678/135746244-f40c8732-72a5-4db1-9356-c0da94a4cb00.png)

## Algorithm

![image](https://user-images.githubusercontent.com/24456678/135746281-3a93628a-942a-4133-962b-66d48e51fb85.png)

*Taken from [Udacity Deep Reinforcement Learning Nanodegree Course](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)*

## Implementation

This code is a slightly modified implementation of the "Lunar Lander" tutorial from Udacity Nanodegree course. It is composed of the following elements:

- model.py: a PyTorch QNetwork class implementation. It initializes a Deep Neural Network composed of:
  -   an input layer (size depending on the state_size parameter)
  -   two hidden layers (units as parameters) 
  -   one output layer (size depending on the action_size parameter)
- dqn_agent.py: DQN Agent Class and a Replay Buffer class implementation
  - DQN Agent implements the following methods: 
    -  constructor: initialization of the Replay Buffer and 2 instance of the Neural Network : the target network and the local network
    - step() allows to store a step taken by the agent (state, action, reward, next_state, done) in the Replay Buffer/Memory. Every 4 steps, it updates the target network weights with the current weight values from the local network
    - act() returns actions for the given state as per current policy using an Epsilon-greedy selection, providing a balance between exploration and exploitation for the Q Learning
    - learn() updates the Neural Network value parameters using batch of experiences from the Replay Buffer
    - soft_update() is called by learn() to softly updates the value from the target Neural Network from the local network weights
  - The ReplayBuffer class implements a fixed-size buffer to store experience tuples (state, action, reward, next_state, done)
    - add() adds an experience step to the memory
    - sample() randomly samples a batch of experience steps for the learning

- DQN_Banana_Navigation.ipynb :
  - Start the environment, train the agent, export the weights and visualize the results

## Hyperparameters

![image](https://user-images.githubusercontent.com/24456678/135748835-6714d0ef-cdbc-457b-a5ba-9874817f253e.png)

## Network architecture

The Neural Networks are composed of one input layer (state space = 37), 2 fully connected hidden layers (size 64, ReLu activation), and an output layer (action space = 4).

## Training

The agent was successfully trained in 419 episodes, achieving an average score of 13 for 100 episodes (between episodes 419 and 519 episodes). 
This successfully meet the objectives of the project instructions (achieving an average score of at least 13 over 100 episodes). 

![image](https://user-images.githubusercontent.com/24456678/135748226-bc28e9a3-1f7d-4a75-b978-8795f43c1a56.png)

![image](https://user-images.githubusercontent.com/24456678/135748278-2c08ebc9-61f0-4044-a5b1-8ba747138cc5.png)


# Ideas for improvement

As stated in the Udacity Nanodegree course, a few improvements could be made:

- [Double Q-Learning](https://arxiv.org/abs/1509.06461)

*Abstract from the paper*
> The popular Q-learning algorithm is known to overestimate action values under certain conditions. [...]
> It was not previously known whether, in practice, such overestimations are common, whether they harm performance, and whether they can generally be prevented. 
> In particular, we first show that the recent DQN algorithm, which combines Q-learning with a deep neural network, suffers from substantial overestimations in some games in the Atari 2600 domain. 
> We then show that the idea behind the Double Q-learning algorithm, which was introduced in a tabular setting, can be generalized to work with large-scale function approximation. 
> We propose a specific adaptation to the DQN algorithm and show that the resulting algorithm not only reduces the observed overestimations, as hypothesized, but that this also leads to much better performance on several games.

From udacity Nanodegree course:

![image](https://user-images.githubusercontent.com/24456678/135748536-346be61a-a129-4beb-8847-ba0ddaa06606.png)
![image](https://user-images.githubusercontent.com/24456678/135748541-89fb4abc-1a45-4500-9ca7-05b40917c61e.png)

- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

*Abstract from the paper*
> Experience replay lets online reinforcement learning agents remember and reuse experiences from the past. In prior work, experience transitions were uniformly sampled from a replay memory. However, this approach simply replays transitions at the same frequency that they were originally experienced, regardless of their significance. 
> In this paper we develop a framework for prioritizing experience, so as to replay important transitions more frequently, and therefore learn more efficiently. We use prioritized experience replay in Deep Q-Networks (DQN), a reinforcement learning algorithm that achieved human-level performance across many Atari games.
> DQN with prioritized experience replay achieves a new state-of-the-art, outperforming DQN with uniform replay on 41 out of 49 games.

![image](https://user-images.githubusercontent.com/24456678/135748578-f2aa510c-f24f-4dae-a229-3d740de278da.png)

- [Dueling Q-Networks](https://arxiv.org/abs/1511.06581)

*Abstract from the paper*
> In recent years there have been many successes of using deep representations in reinforcement learning. Still, many of these applications use conventional architectures, such as convolutional networks, LSTMs, or auto-encoders. In this paper, we present a new neural network architecture for model-free reinforcement learning. 
> Our dueling network represents two separate estimators: one for the state value function and one for the state-dependent action advantage function. The main benefit of this factoring is to generalize learning across actions without imposing any change to the underlying reinforcement learning algorithm. 
> Our results show that this architecture leads to better policy evaluation in the presence of many similar-valued actions. Moreover, the dueling architecture enables our RL agent to outperform the state-of-the-art on the Atari 2600 domain.

![image](https://user-images.githubusercontent.com/24456678/135748610-42ada37f-968a-417b-85f4-13ad895a3e3d.png)

More improvements exist (Learning from multi-step bootstrap targets, Distributional DQN, Noisy DQN).
By incorporating all 6 of them, the research team at Google DeepMind created the [Rainbow algorithm](https://arxiv.org/abs/1710.02298), outperforming all the previous individual modifications.

![image](https://user-images.githubusercontent.com/24456678/135749408-2f01cde9-c897-449f-8c7d-181c68d461f8.png)

