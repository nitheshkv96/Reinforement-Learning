import gym
import torch as T
import numpy as np
import os

from agent import PolicyNetwork, PolicyGradientAgent
import matplotlib.pyplot as plt


# os.makedirs('plots')

def plot_learning_curve(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0,i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running Average pf Previous 100 scores')
    plt.savefig(figure_file)

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 3000
    agent = PolicyGradientAgent(gamma = 0.99, lr = 0.0005, input_dims = [8], n_actions = 4 )
    fname = 'REINFORCE_' + 'LunarLander_lr' + str(agent.lr) + '_'\
                + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    scores = []
    for i in range(n_games):
        done = False
        obs = env.reset()[0]
        score = 0
        while not done:
            action = agent.choose_action(obs)

            obs_, reward, done, info, _ = env.step(action)
            agent.store_rewards(reward)
            score += reward

            obs = obs_
        agent.learn()
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print(f" Episode: {i}, Score: {score:.2f}, average_score:{avg_score:2f}")
        
    x = [i+1 for i in range(len(score))]
    plot_learning_curve(scores, x, figure_file = figure_file)
