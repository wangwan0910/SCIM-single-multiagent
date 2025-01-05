# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 09:54:41 2024
@author: ww1a23
"""

 
import time
import ray  
import argparse
from ray import air  
from ray import tune  
from ray.tune.logger import pretty_print  
from ray.rllib.algorithms.ppo import PPOConfig  
from ray.rllib.algorithms.dqn import DQNConfig  
import gymnasium  
from gymnasium.spaces import Discrete,MultiDiscrete   
import os  
from gymnasium.utils import seeding   
import numpy as np   
import logging   
from datetime import datetime  
import logging  
import numpy as np  
from sgymnasi import (POSingAgent1W1F,POSAgent1W1F_V1T1,POSAgent1W1F_V1T2,POSAgent1W1F_V1T3,POSAgent1W1F_V1T4,
                      POSAgent1W1F_V2T1,POSAgent1W1F_V2T2,POSAgent1W1F_V2T3,
                      POSAgent1W1F_V3T1,POSAgent1W1F_V3T2,POSAgent1W1F_V3T3)
from ray.tune.schedulers import ASHAScheduler  
from ray.tune.stopper import (CombinedStopper,  
                              MaximumIterationStopper,  
                              ExperimentPlateauStopper)  
import ray.rllib.algorithms.ppo as ppo  
from ray.rllib.utils import try_import_torch   

logger = logging.getLogger(__name__)   

# env = GymEnvironment()  

from ray.tune.logger import pretty_print 
import os 
from datetime import datetime 
import matplotlib.pyplot as plt 
import ray 
from ray import tune 
from ray.tune.schedulers import ASHAScheduler 
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper, ExperimentPlateauStopper 
import logging 
import numpy as np 
# from scenv0703 import SupplyChainEnvironment, Action, State 
# from plotrewards0803 import visualize_rewards 
from ray.rllib.env.multi_agent_env import MultiAgentEnv 
from ray.tune.registry import register_env 
from ray.rllib.algorithms.ppo import PPO 
# Register the environment if not already registered 
import torch 
from ray.rllib.policy.sample_batch import SampleBatch 
import numpy as np 


 
 

def trial_dirname_creator(trial): 
    return f"trial_{trial.trial_id}" 

 
 

def train_agent(config, verbose, num_episodes_ray, grace_period_ray, std_episodes_ray, top_episodes_ray, 
                      local_dir, ray_dir): 
    ray.shutdown() 
    ray.init(log_to_driver=False) 
    print('num_episodes_ray', num_episodes_ray) 
    analysis = tune.run( 
        "PPO", 
        config=config, 
        metric='episode_reward_mean', 
        mode='max', 
        scheduler=ASHAScheduler( 
            time_attr='episodes_total', 
            max_t=num_episodes_ray, 
            grace_period=grace_period_ray, 
            reduction_factor=5), 
        stop=CombinedStopper( 
            ExperimentPlateauStopper( 
                metric='episode_reward_mean', 
                patience=5), 
            MaximumIterationStopper(max_iter=num_episodes_ray) 
        ), 
        checkpoint_freq=5, 
        keep_checkpoints_num=5, 
        checkpoint_score_attr='episode_reward_mean', 
        max_failures=5, 
        verbose=verbose, 
        trial_dirname_creator=trial_dirname_creator, 
        storage_path=os.getcwd() + '/' + local_dir + '/' + ray_dir 
    ) 

    results_df = analysis.dataframe() 
    best_result_df = results_df.sort_values('episode_reward_mean', ascending=False).iloc[0] 
    best_config = analysis.best_config 
    best_checkpoint = analysis.best_checkpoint 
    print(f'\n checkpoint saved at {best_checkpoint}') 
    ray.shutdown() 
    return results_df, best_result_df, best_config, best_checkpoint 

 
 
 

  
 

def parse_args():
    parser = argparse.ArgumentParser(description="POSAgent1W1F Environment Configuration")
    parser.add_argument('--env1', action='store_true', help='Use POSAgent1W1F_V1T1 environment')
    parser.add_argument('--env2', action='store_true', help='Use POSAgent1W1F_V1T2 environment')
    parser.add_argument('--env3', action='store_true', help='Use POSAgent1W1F_V1T3 environment')
    parser.add_argument('--env10', action='store_true', help='Use POSAgent1W1F_V1T4 environment')
    parser.add_argument('--env4', action='store_true', help='Use POSAgent1W1F_V2T1 environment')
    parser.add_argument('--env5', action='store_true', help='Use POSAgent1W1F_V2T2 environment')
    parser.add_argument('--env6', action='store_true', help='Use POSAgent1W1F_V2T3 environment')
    parser.add_argument('--env7', action='store_true', help='Use POSAgent1W1F_V3T1 environment')
    parser.add_argument('--env8', action='store_true', help='Use POSAgent1W1F_V3T2 environment')
    parser.add_argument('--env9', action='store_true', help='Use POSAgent1W1F_V3T3 environment')
    return parser.parse_args() 
 

if __name__ == '__main__': 
    
    # env_instance = GymEnvironment() 
    NUM_EPISODES = 250 
    num_episodes_ray = 100000 
    grace_period_ray = num_episodes_ray / 10 
    std_episodes_ray = 5 
    top_episodes_ray = NUM_EPISODES 
    SEED = 2023 
    
    plots_dir = 'plots' 
    now = datetime.now() 
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S") 
    ray_dir = 'ray_results' 
    
 
    std_episodes_ray = 5.0 
  
    
    args = parse_args()
    
    if args.env1:
        envs = POSAgent1W1F_V1T1
    elif args.env2:
        envs = POSAgent1W1F_V1T2
    elif args.env3:
        envs = POSAgent1W1F_V1T3
    elif args.env10:
        envs = POSAgent1W1F_V1T4
    elif args.env4:
        envs = POSAgent1W1F_V2T1
    elif args.env5:
        envs = POSAgent1W1F_V2T2
    elif args.env6:
        envs = POSAgent1W1F_V2T3
    elif args.env7:
        envs = POSAgent1W1F_V3T1
    elif args.env8:
        envs = POSAgent1W1F_V3T2
    elif args.env9:
        envs = POSAgent1W1F_V3T3
    else:
        raise ValueError("Please specify one of --env1, ..., --env10")


    env_instance = envs() 
    local_dir = f"D{env_instance.dist}B{env_instance.b}_S{env_instance.s}_V{env_instance.v_f}T{env_instance.task}_{now_str}" 
    def env_creator(env_config): 
        return envs() 
    register_env("supply_chain_env", env_creator) 


    config = { 
        "env": "supply_chain_env", 
        "env_config": {}, 
        # "multiagent": { 
        #     "policies": { 
        #         "policy1": (None, env_instance.observation_space, env_instance.action_space, { 
        #             "gamma": 0.99, 
        #             "model": { 
        #                 "fcnet_hiddens": [256, 256], 
        #                 "fcnet_activation": "relu", 
        #             }, 
        #             "lr": 0.001, 
        #         }), 
        #         "policy2": (None, env_instance.observation_space, env_instance.action_space, { 
        #             "gamma": 0.95, 
        #             "model": { 
        #                 "fcnet_hiddens": [128, 128], 
        #                 "fcnet_activation": "tanh", 
        #             }, 
        #             "lr": 0.0001, 
        #         }), 
        #     }, 
        #     "policy_mapping_fn": lambda agent_id, episode, **kwargs: "policy1" if agent_id == "warehouse" else "policy2", 
        # }, 
        "framework": "torch", 
        "seed": 2023, 
        "log_level": "DEBUG", 
        "horizon": env_instance.days- 1, 
        "batch_mode": "complete_episodes", 
        "lr": 0.0001, 
        "gamma": 0.99, 
        "rollout_fragment_length": 20, 
        "train_batch_size": 4000, 
        "num_sgd_iter": 10, 
        "sgd_minibatch_size": 512, 
    } 

 
 

    # logger = logging.getLogger('LOGGING_SCIMAI_GYM_VS') 
    logger.setLevel((logging.INFO)) 
    VERBOSE = 3 if logger.level == 10 else 0 

    # Training 
    start = time.time()
    results_df, best_result, best_config, checkpoint = train_agent( 
        config, VERBOSE, num_episodes_ray, grace_period_ray, std_episodes_ray, top_episodes_ray, local_dir, ray_dir 
    ) 
    total_time_taken = time.time()-start
  
    print(f"Total time taken: {total_time_taken/60:.2f} minutes")
    print('results_df',results_df)
    print('best_result',best_result)
    print('best_config',best_config)
    print("checkpoint",checkpoint)
    ray.shutdown() 
 

    # visualize_demand(env_instance, NUM_EPISODES) 

    # save_env_settings(env_instance) 

    # result_df_as_image(best_result, 'PPO') 

    # visualize_rewards(results_df, 'PPO') 

    # def env_creator(env_config): 

    #     return SupplyChainEnvironment() 

    # register_env("supply_chain_env", env_creator) 

    # ray.init(log_to_driver=False) 

    # ppo_trainer = PPO(config=config) 

    # ppo_trainer.restore(checkpoint) 

    # env = env_creator({}) 

    # state = env.reset() 

    # done = False 

    # cumulative_reward = 0 

    # # def policy_mapping_fn(agent_id): 

    # #     if agent_id == "warehouse": 

    # #         return "policy1" 

    # #     elif agent_id == "retailer": 

    # #         return "policy2" 

    # #     else: 

    # #         return "default_policy" 

    # while not done: 

    #     if isinstance(state, tuple) and isinstance(state[0], dict): 

    #         observation_dict, _ = state 

    #     elif isinstance(state, dict): 

    #         observation_dict = state 

    #     else: 

    #         raise ValueError( 

    #             f"Expected state to be a tuple with a dictionary or a dictionary, but got {type(state)}: {state}") 

    #     actions = {} 

    #     # for agent_id, observation in observation_dict.items(): 

    #     #     observation_array = np.array(observation, dtype=np.float32) 

    #     #     input_dict = torch.tensor(observation_array) 

    #     #     policy_id = policy_mapping_fn(agent_id) 

    #     #     action, _, _ = ppo_trainer.get_policy(policy_id).compute_single_action(input_dict, explore=False) 

    #     #     actions[agent_id] = action 

    #     # step_output = env.step(actions) 

    #     # print(f"step_output: {step_output}") 

    #     # if not (isinstance(step_output, tuple) and len(step_output) == 5): 

    #     #     raise ValueError( 

    #     #         f"Expected step_output to be a tuple of length 5, but got {type(step_output)}: {step_output}") 

    #     # state, rewards, dones, infos, _ = step_output 

 
 

    #     # # 检查 state 是否是期望的类型 

    #     # if isinstance(state, dict): 

    #     #     state = (state, None)  # 将字典包装成元组 

 
 

    #     # cumulative_reward += sum(rewards.values()) 

 
 

        # if dones['__all__']: 

        #     done = True 

 
 

    # print("Cumulative reward:", cumulative_reward) 

    

 
 
 
 

# a=gym_env.action_space.sample()  

 
 

# s,r,d,_ = gym_env.step(a)  

 
 

# print(s)  

 
 

# NUM_EPISODES = 250  

 
 

# # number of episodes for RLib agents  

 
 

# num_episodes_ray = 75000  

 
 

# # stop trials at least from this number of episodes  

 
 

# grace_period_ray = num_episodes_ray / 10  

 
 

# # number of episodes to consider  

 
 

# std_episodes_ray = 5.0  

 
 

# # number of epochs to wait for a change in the episodes  

 
 

# top_episodes_ray = NUM_EPISODES  

 
 

# # dir for saving Ray results  

 
 

# ray_dir = 'ray_results'  

 
 

# # name of the experiment (e.g., '2P2W' stands for two product types and two  

 
 

# # distribution warehouses)  

 
 

# now = datetime.now()  

 
 

# now_str = now.strftime('%Y-%m-%d_%H-%M-%S')  

 
 

# local_dir = f"{env.s}V{env.v_f}T{env.task}W_{now_str}"  

 
 

# # dir to save plots  

 
 

# plots_dir = 'plots'  

 
 

# # creating necessary dirs  

 
 

# if not os.path.exists(f"{local_dir}"):  

 
 

#     os.makedirs(f"{local_dir}")  

 
 

# if not os.path.exists(f"{local_dir+'/'+plots_dir}"):  

 
 

#     os.makedirs(f"{local_dir+'/'+plots_dir}")  

 
 

# # creating necessary dir  

 
 

# if not os.path.exists(f"{local_dir+'/'+ray_dir}"):  

 
 

#     os.makedirs(f"{local_dir+'/'+ray_dir}")  

 
 

   

 
 

# VERBOSE = 1  

 
 

   

 
 

# algorithms = {  

 
 

#     'PPO': ppo.PPO,  

 
 

# }  

 
 

   

 
 

   

 
 

   

 
 

   

 
 

# def train(algorithm, config, verbose,  

 
 

#           num_episodes_ray=num_episodes_ray, grace_period_ray=grace_period_ray,  

 
 

#           std_episodes_ray=std_episodes_ray, top_episodes_ray=top_episodes_ray,  

 
 

#           local_dir=local_dir, ray_dir=ray_dir):  

 
 

   

 
 

#     ray.shutdown()  

 
 

#     ray.init(log_to_driver=False)  

 
 

   

 
 

#     analysis = tune.run(algorithm,  

 
 

#                         config=config,  

 
 

#                         metric='episode_reward_mean',  

 
 

#                         mode='max',  

 
 

#                         scheduler=ASHAScheduler(  

 
 

#                             time_attr='episodes_total',  

 
 

#                             max_t=num_episodes_ray,  

 
 

#                             grace_period=grace_period_ray,  

 
 

#                             reduction_factor=5),  

 
 

#                         stop=CombinedStopper(  

 
 

#                             ExperimentPlateauStopper(  

 
 

#                                 metric='episode_reward_mean',  

 
 

#                                 std=std_episodes_ray,  

 
 

#                                 top=top_episodes_ray,  

 
 

#                                 mode='max',  

 
 

#                                 patience=5),  

 
 

#                             MaximumIterationStopper(  

 
 

#                                 max_iter=num_episodes_ray)  

 
 

#                         ),  

 
 

#                         checkpoint_freq=1,  

 
 

#                         keep_checkpoints_num=1,  

 
 

#                         checkpoint_score_attr='episode_reward_mean',  

 
 

#                         # progress_reporter=tune.JupyterNotebookReporter(  

 
 

#                         #     overwrite=True),  

 
 

#                         max_failures=5,  

 
 

#                         verbose=verbose,  

 
 

#                         local_dir=os.getcwd()+'/'+local_dir+'/'+ray_dir)  

 
 

   

 
 

#     trial_dataframes = analysis.trial_dataframes  

 
 

#     best_result_df = analysis.best_result_df  

 
 

#     best_config = analysis.best_config  

 
 

#     best_checkpoint = analysis.best_checkpoint._local_path  

 
 

#     print(f"\ncheckpoint saved at {best_checkpoint}")  

 
 

   

 
 

#     # stopping Ray  

 
 

#     ray.shutdown()  

 
 

   

 
 

#     return trial_dataframes, best_result_df, best_config, best_checkpoint  

 
 

   

 
 

# # https://docs.ray.io/en/master/rllib-algorithms.html#ppo  

 
 

   

 
 

   

 
 

   

 
 

   

 
 

# config_PPO = ppo.copy()  

 
 

# config_PPO['seed'] = 2024  

 
 

# config_PPO['log_level'] = 'WARN'  

 
 

   

 
 

# config_PPO['env'] = GymEnvironment  

 
 

# config_PPO['horizon'] = env.days  

 
 

# config_PPO['batch_mode'] = 'complete_episodes'  

 
 

   

 
 

# config_PPO['model']['fcnet_hiddens'] = tune.grid_search([[64, 64]])  

 
 

# config_PPO['model']['fcnet_activation'] = tune.grid_search(['relu'])  

 
 

   

 
 

# config_PPO['lr'] = tune.grid_search([5e-5])  

 
 

# config_PPO['gamma'] = .99  

 
 

   

 
 

# config_PPO['rollout_fragment_length'] = tune.grid_search(['auto'])  

 
 

# config_PPO['train_batch_size'] = tune.grid_search([8000])  

 
 

   

 
 

# config_PPO['num_sgd_iter'] = tune.grid_search([15])  

 
 

# config_PPO['sgd_minibatch_size'] = tune.grid_search([512])  

 
 

   

 
 

# config_PPO['framework'] = 'torch'  

 
 

   

 
 

# # training a PPO agent  

 
 

# (results_PPO, best_result_PPO,  

 
 

# best_config_PPO, checkpoint_PPO) = train(algorithms['PPO'],  

 
 

#                                           config_PPO,  

 
 

#                                           VERBOSE)  

 
 

                                            

 
 

# # config = (PPOConfig().environment(GymEnvironment).rollouts(num_rollout_workers=2,create_env_on_local_worker=True))  

 
 

# # pretty_print(config.to_dict())  

 
 

# # algo = config.build()  

 
 

# # for i in range(10):  

 
 

# #     result = algo.train()  

 
 

# # print(pretty_print(result))  

 