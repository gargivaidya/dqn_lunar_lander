import gym
from stable_baselines import DQN
import imageio
import numpy as np
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv

# Create environment
env = gym.make('LunarLander-v2')
# Load the trained agent
model = DQN.load("TrainedAgents\dqn_lunar_12")
# Average reward evaluate function
def evaluate(model, num_steps=1000):
  episode_rewards = [0.0]
  obs = env.reset()
  for i in range(num_steps):
      # _states are only useful when using LSTM policies
      action, _states = model.predict(obs)

      obs, reward, done, info = env.step(action)
      
      # Stats
      episode_rewards[-1] += reward
      if done:
          obs = env.reset()
          episode_rewards.append(0.0)
  # Compute mean reward for the last 100 episodes
  mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
  print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
  
  return mean_100ep_reward

# Evaluate the agent
mean_reward = evaluate(model, num_steps=1000)

# Enjoy trained agent
images = []
obs = env.reset()
img = env.render(mode='rgb_array')
for i in range(300):
    images.append(img)
    action, _ = model.predict(obs)
    obs, _, _ ,_ = env.step(action)
    img = env.render(mode='rgb_array')

imageio.mimsave('Evaluation gifs\lander_dqn_12_eval_2.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)