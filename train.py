# @author: Ahmet Furkan DEMIR

# Modules
from wrappers import wrap_mario
import configPy
import ray
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.models import ModelCatalog
from tabulate import tabulate
from ray.tune.registry import register_env
from TFModel import TFQModel

# print_results (DQN)
def print_results(result, iteration):
    table = [['DQN',
              iteration,
              result['timesteps_total'],
              round(result['episode_reward_max'], 3),
              round(result['episode_reward_min'], 3),
              round(result['episode_reward_mean'], 3)]]
    print(tabulate(table,
                   headers=['Agent',
                            'Iteration',
                            'Steps',
                            'Max Reward',
                            'Min Reward',
                            'Mean Reward'],
                   tablefmt='psql',
                   showindex="never"))
    print()

# init
ray.init()

def env_creator_lambda(config):
  return wrap_mario()

# register env (gym-super-mario-bros)
register_env('super_mario_bros', env_creator_lambda)

# get config (configPy.py)
config = configPy.getConfig()

#register model
ModelCatalog.register_custom_model('TFModel', TFQModel)

# DQN
trainer = DQNTrainer(config=config)

# Print config
print(trainer.get_config())
# Print model
policy = trainer.get_policy()
model = policy.model
print(model.base_model.summary())

for i in range(500000):
  
   # Perform one iteration of training the policy with DQN
   result = trainer.train()
   print_results(result, i)

   if i % 300 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)