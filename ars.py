## Augmented random Search #####

#### This is a basic Psuedo-code to Implementation of  Random Search AI#####
#### This does NOT include Contributions to it for improvements using Normalization ###

###Pull Requests for improvements on the code and further contributions is welcomed ###


# Importing the required libraries
import os
import numpy as np
import gym
from gym import wrappers
import pybullet_envs           ##custom pybullet Simulation library

# Setting up the Hyper Parameters

class Hyperparameters():

    def __init__(self):
        self.nb_steps = 1000           ##number of steps
        self.episode_length = 1000     ## maximum length o feach possible episode
        self.learning_rate = 0.02      ## basic learning rate
        self.nb_directions = 16        ##total possible perturbations
        self.nb_best_directions = 16   ##total best possible perturbations
        ### chekcing for the number of best directions to be less than total directions
        assert self.nb_best_directions <= self.nb_directions
        ## hyperparameters to be considered according to the research paper
        self.noise = 0.03
        self.seed = 1
        self.env_name = 'HalfCheetahBulletEnv-v0'



# Building the policy which will decide the locomotion of half-Cheetah

class Policy():

    def __init__(self, input_size, output_size):
        self.theta = np.zeros((output_size, input_size))

    def evaluate(self, input, delta = None, direction = None):        ###Perceptrons
        if direction is None:
            return self.theta.dot(input)
        elif direction == "positive":
            return (self.theta + hp.noise*delta).dot(input)
        else:
            return (self.theta - hp.noise*delta).dot(input)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]

    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step

## Building function for exploring the policy on one specific direction and over one episode
   # - Reset the state
   # - Boolean variables to identify the end of each episode
   # - variable for number of action played (Ini=0)
   # - variable for accumulated reward (Ini=0)
   # - Loop untill number of action played reaches end of Loop
   # - return a normlaized state
   # - feed the above result to the Perceptrons
   # - Update the pyBullet Env object , Step() method

   ## To remove the bias and outliers of very high positive and negative rewards
   ### we use the classic trick of reinforcement learning
       # setting +1 for very high positive reward
       # setting -1 for very high negative rewards


# Exploring the policy on one specific direction and over one episode

def explore(env, normalizer, policy, direction = None, delta = None):
    state = env.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0
    while not done and num_plays < hp.episode_length:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        reward = max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1
    return sum_rewards       


# Training the model

def train(env, policy, normalizer, hp):

    for step in range(hp.nb_steps):

        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions

        # Getting the positive rewards in the positive directions
        for k in range(hp.nb_directions):
            positive_rewards[k] = #explore(env, normalizer, policy, direction = "positive", delta = deltas[k])

        # Getting the negative rewards in the negative/opposite directions
        for k in range(hp.nb_directions):
            negative_rewards[k] = # explore(env, normalizer, policy, direction = "negative", delta = deltas[k])

        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()

        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key = lambda x:scores[x], reverse = True)[:hp.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

        # Updating our policy
        policy.update(rollouts, sigma_r)

        # Printing the final reward of the policy after the update
        reward_evaluation = #explore(env, normalizer, policy)
        print('Step:', step, 'Reward:', reward_evaluation)

# Running the main code

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir('result', 'ars')
simulation_dir = mkdir(work_dir, 'simulation')

hp = Hyperparameters()
np.random.seed(hp.seed)
env = gym.make(hp.env_name)
env = wrappers.Monitor(env, monitor_dir, force = True)
nb_inputs = env.observation_space.shape[0]
nb_outputs = env.action_space.shape[0]
policy = Policy(nb_inputs, nb_outputs)
normalizer = Normalizer(nb_inputs)
train(env, policy, normalizer, hp)
