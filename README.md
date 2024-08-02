# Reinforcement-Learning
This repository provides base code for various Reinforcement Learning Algorithms and beer game simulation using RL.

## Supervised Unsupervised algorithms vs RL
* Supervised & Unsupervised Machine Learning - these are static tasks where we have the input dataset and a deterministic output that if we choose these variables then this particular output will be received. Example- object classification, clustering
* Reinforcement Learning - this is a dynamic tasks where we have to take steps or perform actions and experience to know the output. Example- human strategy like chess, tic tac toe
<br><br>

## Reinforcement Learning
<b>Reinforcement Learning (RL)</b> refers to a kind of Machine Learning method in which the agent acts on its environment, it receives some evaluation of its action (reinforcement), but is not told of which action is the correct one to achieve its goal.<br><br>
Example-<br>
<b>Game playing:</b> player knows whether it win or lose, but not know how to move at each step<br>
<b>Goal:</b> Learn how to take actions in order to maximize reward.<br>
<br><br>
Typically, a RL setup is composed of two components, an agent and an environment.
![image](https://github.com/user-attachments/assets/8e09978b-71e9-47af-9130-7206dd8b37e3)

The environment starts by sending a state to the agent, which then based on its knowledge take an action in response to that state. <br>
After that, the environment send a pair of next state and reward back to the agent. The agent will update its knowledge with the reward returned by the environment to evaluate its last action.<br> 
The loop keeps going on until the environment sends a terminal state (win/loss/draw), which ends to episode (a single game).<br><br>

<b>Key terms in RL-</b>
1. Agent: Thing that senses the environment and which we are trying to make intelligent.
2. Environment: Real world or simulated world that agent lives in example- chess board in game chess
3. Action (A): All the possible moves that the agent can take in the environment
4. State (S): Different configurations of the environment that agent can sense
5. Reward (R): An immediate return from the environment to evaluate the last action of agent
6. Policy (π): The strategy that the agent employs to determine next action based on the current state or reward that it received from environment
7. Value (V): The expected long-term return with discount, as opposed to the short-term reward R. Vπ(s) is defined as the expected long-term return of the current state under policy π.
8. Q-value or action-value (Q): Q-value is similar to Value, except that it takes an extra parameter, the current action a. Qπ(s, a) refers to the long-term return of the current state s, taking action a under policy π.<br>
<br>
<b>Steps for a complex reinforcement learning-</b>
1. Agent observes an input state
2. An action is determined by decision making function (policy)
3. The action is performed
4. The agent receives a scalar reward from the environment
5. Information about the reward received and action performed are recorded
<br>

Various Methods to approach a Reinforcement Learning Solution-
	1. Markov Decision Process (MDP) - 
	A probabilistic model of a sequential decision problem, where states can be perceived exactly, and the current state and action selected determine a probability distribution on future states. Essentially, the outcome of applying an action to a state depends only on the current action and state (and not on preceding actions or states).
	2. Dynamic Programming (DP) -
	is a class of solution methods for solving sequential decision problems with a compositional cost structure.
	3. Monte Carlo methods -
	A class of methods for learning of value functions, which estimates the value of a state by running many trials starting at that state, then averages the total rewards received on those trials.
	4. Temporal Difference Algorithm (TD) -
	A class of learning methods, based on the idea of comparing temporally successive predictions. Possibly the single most fundamental idea in all of reinforcement learning.
	
	
	
	A non-exhaustive taxonomy of RL algorithms

Model-Free vs Model-Based Algo -
Model-based learning attempts to model the environment then choose the optimal policy based on it’s learned model; In Model-free learning the agent relies on trial-and-error experience for setting up the optimal policy.


Q Learning-
The goal is to maximize the Q-value. Q-learning belongs to the off-policy category i.e. the next action is chosen to maximize the next state’s Q-value instead of following the current policy.
Its main weakness is lack of generality. In other words, Q-learning agent does not have the ability to estimate value for unseen states.

State-Action-Reward-State-Action (SARSA)-
SARSA very much resembles Q-learning. The key difference between SARSA and Q-learning is that SARSA is an on-policy algorithm. It implies that SARSA learns the Q-value based on the action performed by the current policy instead of the greedy policy.
This also comes with the limitation of generality.

Deep Q Network (DQN)-
DQN leverages a Neural Network to estimate the Q-value function to overcome the Q learning limitation.
Cons- can only handle discrete, low-dimensional action spaces.

Deep Deterministic Policy Gradient (DDPG)-
a model-free, off-policy, actor-critic algorithm that tackles the limitation of DQN by learning policies in high dimensional, continuous action spaces



When Not to Use Reinforcement Learning?
You can't apply reinforcement learning model is all the situation. Here are some conditions when you should not use reinforcement learning model.
	1. Reinforcement learning is not preferable to use for solving simple problems
	2. When you have enough data to solve the problem with a supervised learning method
	3. The curse of dimensionality limits reinforcement learning heavily for real physical systems
	4. Reinforcement Learning is computing-heavy and time-consuming. In particular when the action space is large

Why use Reinforcement Learning?
Here are prime reasons for using Reinforcement Learning:
	1. Maximizes Performance
	2. Sustain Change for a long period of time i.e. this technique is preferred to achieve long-term results which are very difficult to achieve.
	3. It can create the perfect model to solve a particular problem as it can correct errors in training process with very less chances of repeating them.
	4. In the absence of a training dataset, it is bound to learn from its experience.
	5. Reinforcement learning is intended to achieve the ideal behavior of a model within a specific context, to maximize its performance.
	6. Reinforcement learning algorithms maintain a balance between exploration and exploitation. Exploration is the process of trying different things to see if they are better than what has been tried before. Exploitation is the process of trying the things that have worked best in the past. Other learning algorithms do not perform this balance.

Challenges of Reinforcement Learning
Here are the major challenges you will face while doing Reinforcement earning:
	1. Reward design which should be very involved
	2. Parameters (no. of states, learning rate, no. of actions) may affect the speed of learning.
	3. Realistic environments can have partial observability.
	4. Too much Reinforcement may lead to an overload of states which can diminish the results.
	5. Realistic environments can be non-stationary.

Applications-
	1. Business strategy planning
	2. industrial automation or for the learning of robots. Robots are trained using the trial and error method with human supervision. Reinforcement learning teaches robots new tasks while retaining prior knowledge
	3. Reinforcement learning can also be applied in optimizing chemical reactions. (new products packaging experiments)


	


![image](https://github.com/user-attachments/assets/c3587bd5-ec3f-4c2c-938e-825a0ab03322)

