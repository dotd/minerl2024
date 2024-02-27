import gymnasium as gym

env = gym.make('HalfCheetah-v4')

obs = env.reset()

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, b1, b2, info = env.step(action)
    print(f"obs.shape={obs.shape}; reward={reward}; b1={b1}; b2={b2}; info={info}")

