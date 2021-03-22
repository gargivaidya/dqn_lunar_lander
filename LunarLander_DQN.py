import gym
from stable_baselines import DQN
import numpy as np
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.deepq.policies import FeedForwardPolicy, register_policy

# Create environment
env = gym.make('LunarLander-v2')

# Create CustomDQNPolicy
class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[128, 128, 128],
                                           layer_norm=True,
                                           feature_extraction="mlp")
# Register Custom DQN Policy
register_policy('CustomDQNPolicy', CustomDQNPolicy)

# Instantiate the agent
model = DQN('CustomDQNPolicy', env, learning_rate=1e-3, 
			buffer_size = 50000, exploration_fraction = 0.1, 
			exploration_final_eps = 0.05, exploration_initial_eps = 1.0,  
			prioritized_replay=True, prioritized_replay_alpha = 0.3, 
			prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
            prioritized_replay_eps=1e-6, verbose=1, 
			tensorboard_log="./dqn_lunar_tensorboard/")

# Train the agent
model.learn(total_timesteps=200000, tb_log_name="mlp_run")
# Save the agent
model.save("TrainedAgents\dqn_lunar_12")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = DQN.load("TrainedAgents\dqn_lunar_12")

# Evaluate the agent
mean_reward = evaluate(model, num_steps=1000)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
