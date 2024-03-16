import gym

if __name__ == '__main__':
    env = gym.make('LunarLander-v2',render_mode="human")

    n_games = 100

    for i in range(n_games):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs_, reward, done, info, _ = env.step(action)
            score += reward
            # env.render()
        print(f" Episode: {i}, Score: {score:.3f}")

