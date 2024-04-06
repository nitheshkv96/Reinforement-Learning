import numpy as np
from network import ActorNetwork, CriticNetwork
import gym
from agent import Agent
import matplotlib.pyplot as plt


def plot_learning_curve(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0,i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running Average pf Previous 100 scores')
    plt.savefig(figure_file)

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.0001, beta=0.001,
                 in_dims=env.observation_space.shape, tau=0.001,
                 batch_size=64, h_dims=[400,300],
                 n_actions=env.action_space.shape[0])
    n_games = 1000
    filename = 'LunarLander'+ '_ddpg'
    figure_file = 'plots/'+ filename +'.png'

    best_score = env.reward_range[0]
    score_history = []

    for i in range(n_games):
        obs = env.reset()
        obs = np.array(obs[0])
        done = False
        score = 0
        agent.noise.reset()

        while not done:
            action = agent.choose_actions(obs)
            obs_, reward, done, info, _ = env.step(action)
            agent.remember(obs, action, reward, obs_, done)
            score += reward
            obs  = obs_
            # env.render()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' %score, 
              'average score %.1f'% avg_score)
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
