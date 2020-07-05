
![Build status](https://ci.appveyor.com/api/projects/status/ugq1vwa8045p307g?svg=true)
![Build Status](https://travis-ci.org/prateekiiest/Code-Sleep-Python.svg?branch=master)
<img src="https://opencollective.com/code-sleep-python/tiers/sponsor/badge.svg?label=sponsor&color=brightgreen" />
<img src="https://opencollective.com/code-sleep-python/tiers/backer/badge.svg?label=backer&color=brightgreen" />
![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![made with &hearts in Python](https://img.shields.io/badge/made%20with%20%E2%9D%A4%20in-Python-red.svg)](http://shields.io/#your-badge)
# Augmented Random Search

 

This project aims to build a new type of Artificial intelligence algorithm which is simple and surpasses many already available algorithms for Humanoid or Mu-Jo-Co(Multidimensionla-Joint-with-contact)
locomotion related tasks. It simulates a powerful AI Algorithm, namely Augmented Random Search (ARS) by training a Half-cheetah (Mu-Jo-Co) to walk and run across a field.
to walk and run .

## Motivation 
[Link](https://www.youtube.com/watch?v=hx_bgoTF7bs) to the Google-DeepMind's Video

## Existing methods
* Asynchronous Actor-Critic Agents
* Deep Learning 
* Deep Reinforcement Learning

## How is it different ?
  * Unlike other AI systems where the exploration occurs after each action (Action Space) , here exploration occurs after end of each [episode](https://www.quora.com/What-does-the-term-%E2%80%9Cepisode%E2%80%9D-mean-in-the-context-of-reinforcement-learning-RL) (Policy space)
  * ARS is a shallow learning technique unlike deep learning in other AI's systems (Uses only one perceptron rather than layers of it)
  * ARS discards the technique of Gradient Descent for weight adjustment and uses the [Method of Finite Differences](https://en.wikipedia.org/wiki/Finite_difference_method)

# Implementation
 ### Components
 * Perceptrons
 * Reward Mechanism and updation of weights
 * Method of finite Differences to find the best possible direction of movement
 
 ### Algorithm 
 * Scaling the update step by standard deviation of Rewards.
 * Online normalization of weights.
 * Choosing better directions for faster learning.
 * Discarding directions that yield lowest rewards.
 
 ### Algorithm Overview
 ![Alt text](https://github.com/ashutoshtiwari13/Simple-Random-Search/blob/master/photos/SS11.png)
 

 ## Installation 
 - Fork and clone the repository using ``` git clone https://github.com/ashutoshtiwari13/Simple-Random-Search.git ```
 - Run ```pip install -r requirements.txt ``` 
 - Also check the Simulation.txt for setting up the PyBullet Simulation Environment
 - Use the Anaconda Cloud - Spyder IDE (Any framework/IDE of your choice)
 - Use Python 3.6 and above
 - Run the command ``` python ars.py ```
 
 # Results
  ### Reference Mu-ju-Co
  ![Alt text](https://github.com/ashutoshtiwari13/Simple-Random-Search/blob/master/photos/SS6.png)
  
  ### Series of Rewards
  Rewards start from being negative as low as -900 and climbs to positive 900 in around 1000 steps.
  ![Alt Text](https://github.com/ashutoshtiwari13/Simple-Random-Search/blob/master/photos/SS5.png)
  ![Alt Text](https://github.com/ashutoshtiwari13/Simple-Random-Search/blob/master/photos/SS4.png)
  ![Alt Text](https://github.com/ashutoshtiwari13/Simple-Random-Search/blob/master/photos/SS3.png)
 
 
 
 # Simulation Images
 ![Alt text](https://github.com/ashutoshtiwari13/Simple-Random-Search/blob/master/photos/SS12.jpg)
 
 # Further reading 
 - Ben Recht's [Blog](http://www.argmin.net/2018/03/20/mujocoloco/)
 - Reference paper - [Link](https://arxiv.org/pdf/1703.03864.pdf)
 - Research paper used - [Link](https://arxiv.org/pdf/1803.07055.pdf)
 
 Happy coding :blush: :heart: :heavy_check_mark:
 
