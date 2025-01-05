# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 09:53:44 2024
@author: ww1a23
"""


from gymnasium.spaces import Discrete,MultiDiscrete   
import datetime  
import os   
import gymnasium   
import csv  
import numpy as np   
import time  
from gymnasium import utils   
from gymnasium.utils import seeding   
import matplotlib.colors as mcolors  
import matplotlib.pyplot as plt  
from scipy.stats import poisson, binom, randint, geom   
import time  
from itertools import chain  
import collections  
import logging   
import datetime
 
 

logger = logging.getLogger(__name__)   

 
ep_count = 0  
class POSingAgent1W1F:   
    def __init__(self, *args, **kwargs):  
        # super().__init__(*args, **kwargs)  
        self.factory_capacity = 60  
        self.retailer_capacity = 20   
        self.price_capacity = 60  
        factory_capacity = self.factory_capacity   
        retailer_capacity = self.retailer_capacity   
        price_capacity = self.price_capacity   
        self.v_f = 2 # 1:the factory does not show its stock; 2:factory can choose to show its stock or not.  
        self.action_space = MultiDiscrete([retailer_capacity + 1,price_capacity+1,self.v_f,retailer_capacity + 1,price_capacity+1,self.v_f], dtype = np.int32)  #[order , price, show]  
        self.periods = 30 #simulation days  
        self.observation_space = MultiDiscrete([factory_capacity,factory_capacity,factory_capacity,factory_capacity+1,factory_capacity,factory_capacity,factory_capacity,self.periods+1,factory_capacity,factory_capacity,factory_capacity,factory_capacity+1,factory_capacity,factory_capacity,factory_capacity,self.periods+1], dtype = np.int32) # [inventory,backlog,stockout,Factory_inventory]  
        self.state =  np.array([10,0,0,0,0,0,0,0,10,0,0,0,0,0,0,0],dtype=np.int32)  
        self.r = {1:0, 2:0}   
        self.info =  self.state  
        self.k = 5   
        self.c = 2   
        self.h = 2   
        self.b = 5  
        self.s = [200,100]  
        self.re_num = 0  
        self.days =  30  
        self.day =  0  
        self.stockoutCount  = 0  
        self.stockoutCountS  = 12  
        self._seed()   
        self.dist = 6  #This parameter can be selected for 1-5 different demand distributions.  
        self.ALPHA = 0.5  
        self.dist_param = {'mu':10,'muDemand':2,'stdDemand':1} 
        self.d_max = 30  
        self.d_var = 5  
        self.task = 1 #1: rewards set remains the same, 2: factory rewards need to be subtracted from retailer rewards, 3: retailer incentives need to be subtracted from factory rewards  
        self.demand_len=3  
        self.reset()   
        self.agent1_action_1_count = 0  
        self.agent1_action_0_count = 0  
        self.agent2_action_1_count = 0  
        self.agent2_action_0_count = 0  
        self.total_steps = 0   
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.demand_history = []



    def demand(self):  
        distributions = {  
            1: {'dist': poisson, 'params': {'mu': self.dist_param['mu']}},  
            2: {'dist': binom, 'params': {'n': 10, 'p': 0.5}},  
            3: {'dist': randint, 'params': {'low': 0, 'high': 50}},  
            4: {'dist': geom, 'params': {'p': 0.5}},  
            5: {'dist': 'seasonal', 'params': None},
            6: {'dist': 'lasc', 'params': {'muDemand': self.dist_param['muDemand'], 'stdDemand': self.dist_param['stdDemand']}}  
        }  
    
        if self.dist in distributions:
            dist_info = distributions[self.dist]  
            dist_instance = dist_info['dist']  
            dist_params = dist_info['params']  
            
            # Handle standard distribution cases (1 to 4)
            if self.dist < 5 and dist_instance is not None:  
                return dist_instance.rvs(**dist_params)  
    
            # Handle seasonal demand (dist == 5)
            elif self.dist == 5:  
                current_date = datetime.datetime.now()  
                t = current_date.timetuple().tm_yday  # Day of the year (1 to 365)
                demand = np.round(  
                    self.d_max / 3 +  
                    self.d_max / 5 * np.cos(4 * np.pi * (2 + t) / self.periods) +  
                    np.random.randint(0, self.d_var + 1)  
                )  
                demand = min(demand, 30) 
                return int(demand)  # Return as an integer
            
            # Handle custom normal distribution (dist == 6)
            elif self.dist == 6:
                muDemand = self.dist_param['muDemand']
                stdDemand = self.dist_param['stdDemand']
                demand = np.floor(max(0, np.random.normal(muDemand, stdDemand)))
                return int(demand)  # Ensure it returns an integer
        else:
            raise ValueError(f"Invalid distribution type: {self.dist}") 
 
 

    def transition(self, x, a_factory, a_retailer,V_F,d):  
        I = [x[0],x[8]]  
        BL = [x[1],x[9]]  
        SO =  [x[2],x[10]]  
        total_inventory_factory = a_factory + I[1]  
        factory_stockout = False  
        if  0 <= total_inventory_factory <= self.factory_capacity:  #### blance  
            I[1] = total_inventory_factory  
            if  I[1] < a_retailer: #### stockout  
                SO[1] = a_retailer-I[1]  
                if SO[1] > 0:  
                    factory_stockout = True  
                    # for agent_id in agent_ids:  
                    self.stockoutCount += 1  
                    a_retailer = I[1] #*********     order changed = real order/demand  
                I[1] = 0  
                BL[1] = 0  
            else:  
                I[1] -= a_retailer  
                SO[1] = 0  
                BL[1] = 0  
        elif total_inventory_factory > self.factory_capacity:  
                BL[1] = total_inventory_factory-self.factory_capacity  
                I[1] = total_inventory_factory  
                if I[1] < a_retailer: #### stockout  
                    SO[1] = a_retailer-I[1]  
                    if SO[1] > 0:  
                        factory_stockout = True  
                        # for agent_id in agent_ids:  
                        self.stockoutCount += 1  
                    I[1] = 0         
                else:  
                    I[1] -= a_retailer  
                    SO[1] = 0  
                    BL[1] = np.where(I[1] < self.factory_capacity, 0, BL[1])   

        I[1] = np.clip(I[1],0, self.factory_capacity-1)  
        BL[1] = np.clip(BL[1],0, self.factory_capacity-1)  
        SO[1] = np.clip(SO[1],0, self.factory_capacity-1)  

        if not factory_stockout:  
            total_inventory_retailer = a_retailer + I[0]  
            if  0 < total_inventory_retailer <= self.retailer_capacity:  
                I[0] = total_inventory_retailer  
                if  I[0] < d:  
                    SO[0] = d-I[0]  
                    if SO[0] > 0:  
                        self.stockoutCount += 1  
                        d = I[0] #*********     order changed = real order/demand  
                    I[0] = 0  
                    BL[0] = 0  
                else:  
                    I[0] -= d  
                    SO[0] = 0  
                    BL[0] = 0  
            elif total_inventory_retailer > self.retailer_capacity:  
                # self.retailer_capacity < a_retailer+I[0]:  
                    BL[0] = total_inventory_retailer-self.retailer_capacity  
                    I[0] = total_inventory_retailer  
                    if d > I[0]:  
                        SO[0] = d-I[0]  
                        if SO[0] > 0:  
                            self.stockoutCount += 1  
                        I[0] = 0  
                    else:  
                        I[0] -= d  
                        SO[0] = 0  
                        BL[0] = np.where(I[0] < self.retailer_capacity, 0, BL[0])  

            I[0] = np.clip(I[0],0, self.retailer_capacity-1)  
            BL[0] = np.clip(BL[0],0, self.retailer_capacity-1)  
            SO[0] = np.clip(SO[0],0, self.retailer_capacity-1)  


        else:  
            a_retailer = 0 #*********     order changed = real order/demand  
            total_inventory_retailer = I[0]  
            if  I[0] < d:  
                SO[0] = d-I[0]  
                if SO[0] > 0:   
                    self.stockoutCount= 1  
                    d = I[0] #*********     order changed = real order/demand  
                I[0] = 0  
                BL[0] = 0  
            else:  
                I[0] -= d  
                SO[0] = 0  
                BL[0] = 0  

 

        self.o_history.append(np.array([[a_retailer]], dtype=np.int64))  
        o_history = np.hstack(list(chain(*chain(*self.o_history))))  
        self.d_history.append(np.array([[d]], dtype=np.int64))  
        d_history = np.hstack(list(chain(*chain(*self.d_history))))  
        self.period +=1  


        x =  np.array([I[0],BL[0],SO[0],d_history[0],d_history[1],d_history[2],I[1]*V_F,self.period,I[1],BL[1],SO[1],o_history[0],o_history[1],o_history[2],0,self.period])  
        return x  

 
 

    def reward(self, x, a_retailer,p_retailer,a_factory, p_factory,d, y):   
        k = self.k   
        c = self.c   
        h = self.h  
        factory_reward = - c * max(min(y[8] + a_factory, self.factory_capacity) - y[8], 0)-h * y[8] + p_factory * max(min(x[8] + x[13], self.factory_capacity) - x[8], 0)   
        if self.task == 2:  
            factory_reward -= self.s[0]*y[2]  
        #self.r[2] = factory_reward  
        self.day+= 1  
        retailer_reward =- p_factory * max(min(y[0] + y[13], self.retailer_capacity) - y[0], 0)-h * y[0]   + p_retailer * max(min(x[0] +x[5], self.retailer_capacity) - x[0], 0)  
       
        if self.task == 3:  
            retailer_reward -= self.s[1]*y[10]  
        
        if self.task == 4:  
            retailer_reward -= self.s[1]*y[10]  
            factory_reward -= self.s[0]*y[2]  
            
        #self.r[1] = retailer_reward 
        
        self.r[1] = -self.s[0]*y[2] * 0.7 -self.b * y[1] * 0.2 + retailer_reward*0.1
        
        self.r[2] = -self.s[1]*y[10] * 0.7 - self.b*y[9] * 0.2 + factory_reward *0.1
        
        r = self.r[1] + self.r[2]  
        return r  

 
 

    def _seed(self, seed=None):   
        self.np_random, seed = seeding.np_random(seed)   
        return [seed]   

 
 

    def step(self, action):   
        global ep_count  
        ep_count += 1  
        obs_dict = self.state.copy()   
        demand = self.demand()   
        action = list(action)

        action[2] = 0
        if self.v_f == 2:
            action[5] = np.where(action[5] >= 1, 1, 0)
        elif self.v_f == 3:
            action[5] = 1
        else:
            action[5] = 0
        self.total_steps += 1  
        observations_dict = self.transition(obs_dict, action[3], action[0],action[5], demand)   
        self.state = observations_dict   
        rewards = self.reward(obs_dict, action[0],action[1],  action[3], action[4], demand,observations_dict)   
        done = self.is_done()  
        self.state_history.append(self.state.copy())
        self.reward_history.append(rewards.copy())
        self.demand_history.append(demand)
        action[1]=""
        self.action_history.append(action)
        
        self.render()
        return observations_dict, rewards, done,done, {}   


    def is_done(self):   
        if self.stockoutCount >= self.stockoutCountS or self.day >= self.days:  
            done = True  
        else:  
            done = False 
        return done   


    def reset(self, seed=None, options=None): 
        self.re_num += 1  
        self.stockoutCount = 0  
        self.d_history = collections.deque(maxlen=self.demand_len)  
        for d in range(self.demand_len):  
            self.d_history.append(np.zeros((1,1), dtype = np.int32))  
        self.o_history = collections.deque(maxlen=self.demand_len)  

        for d in range(self.demand_len):  
            self.o_history.append(np.zeros((1,1), dtype = np.int32))  

        self.t = 0  
        self.day = 0  
        self.period = 0  
        self.r = {1: 0, 2: 0}  
        self.agent1_action_1_count = 0  
        self.agent1_action_0_count = 0  
        self.agent2_action_1_count = 0  
        self.agent2_action_0_count = 0  
        self.total_steps = 0   
        self.state = np.array([10,0,0,0,0,0,0,0,10,0,0,0,0,0,0,0],dtype=np.int32) 


        return self.state ,{} 
    
    def save_to_csv(self, filename=None):
       # filename=f"{self.__class__.__name__}.csv"
       # print('1111111111',self.reward_history) #1111111111 [-224]
       data = []
       for i in range(len(self.state_history)):
           row = {
               "state_1_inventory": self.state_history[i][0],
               "state_1_backlog": self.state_history[i][1],
               "state_1_stockout": self.state_history[i][2],
               "state_1_demand1": self.state_history[i][3],
               "state_1_demand2": self.state_history[i][4],
               "state_1_demand3": self.state_history[i][5],
               "state_1_SHOW_Factory_inv": self.state_history[i][6],
               "state_1_steps": self.state_history[i][7],

               "action_1_order": self.action_history[i][0],
               "action_1_price": self.action_history[i][1],
               "action_1_SHOW": self.action_history[i][2], ### always 0
               
               # "reward_1": self.reward_history[i][0],

               "state_2_inventory": self.state_history[i][8],
               "state_2_backlog": self.state_history[i][9],
               "state_2_stockout": self.state_history[i][10],
               "state_2_order1": self.state_history[i][11],
               "state_2_order2": self.state_history[i][12],
               "state_2_order3": self.state_history[i][13],
               "state_2_SHOW_Factory_inv": self.state_history[i][14],
               "state_2_steps": self.state_history[i][15],

               "action_2_order": self.action_history[i][3],
               "action_2_price": self.action_history[i][4],
               "action_2_SHOW": self.action_history[i][5],  ### 0 or 1
               "reward": self.reward_history[i],  
               # "action_2_SHOW": self.action_history[i][2][2], 
           }
           data.append(row)
       # print(f"Data to be saved: {data}") #Data to be saved: []
       with open(filename, 'w', newline="") as csvfile:
            fieldnames = [
                "state_1_inventory", "state_1_backlog", "state_1_stockout","state_1_demand1","state_1_demand2","state_1_demand3",
                "state_1_SHOW_Factory_inv","state_1_steps",
                "action_1_order",
                "action_1_price",
                "action_1_SHOW",
                "state_2_inventory", "state_2_backlog", "state_2_stockout","state_2_order1","state_2_order2","state_2_order3",
                "state_2_SHOW_Factory_inv","state_2_steps",
                "action_2_order",
                "action_2_price",
                "action_2_SHOW",
                "reward"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    
    
    def render(self, mode="human", log_dir='./plots'):
        current_date = datetime.datetime.now()
        global ep_count
        p = self.periods
        p = 500 if p <= 500 else p
    
        # Ensure that rendering happens every 'p' steps
        if ep_count % p == 0 and len(self.state_history) % p == 0:
            start = max(0, len(self.state_history) - p)
            end = len(self.state_history)
    
            fig = plt.figure(figsize=(10, 10))  # Adjusted figure size
            agent_labels = ["Retailer", "Factory"]  
    
            for i in range(2):
                ax = fig.add_subplot(4, 2, i + 1)
    
                # Plot Demand/Order for Retailer and Factory
                if i == 0:
                    # For Retailer, plot state_1_demand3
                    ax.plot(range(start, end), [s[5] for s in self.state_history[start:end]], label=f'Demand', color='b', alpha=0.85, linewidth=0.8)
                else:
                    # For Factory, plot state_2_order3
                    ax.plot(range(start, end), [s[13] for s in self.state_history[start:end]], label=f'Demand', color='b', alpha=0.85, linewidth=0.8)
                

                # Plot Agent Orders
                ax.plot(range(start, end), [a[i*3] for a in self.action_history[start:end]], label=f'Agent {agent_labels[i]} Order', color='mediumslateblue', alpha=0.85, linewidth=0.8)
                if i == 0:
                    ax.plot(range(start, end), [a[i*3 + 1] for a in self.action_history[start:end]], color="ghostwhite", alpha=0.1, linestyle='--', linewidth=0.1)
                else:
                    mean_price = sum([a[i*3 + 1] for a in self.action_history[start:end]]) / (end - start)
                    ax.plot(range(start, end), [a[i*3 + 1] for a in self.action_history[start:end]], label=f'Agent {agent_labels[i]} Mean Price: {mean_price:.2f}', color="lightsteelblue", alpha=0.4, linestyle='--', linewidth=1.7)
    
                ax.legend(loc='upper left', fontsize='medium', markerscale=0.5)
    
                if i == 0:
                    ax.set_ylabel('Demand & Order & Price', fontsize='medium')
                
                ax.set_ylim([-1, max(self.factory_capacity, self.retailer_capacity) + 1])
    

            for i in range(2):
                ax = fig.add_subplot(4, 2, i + 3)
                inventory_data = [s[i*8] for s in self.state_history[start:end]]
                mean_inventory = np.mean(inventory_data)
                ax.plot(range(start, end), inventory_data, label=f'Mean inventory for Agent {agent_labels[i]}: {mean_inventory}', color='navy', alpha=0.3, linewidth=2)
                ax.plot(range(start, end), [s[i*8 + 1] for s in self.state_history[start:end]], label=f'Agent {agent_labels[i]} Backlog', color='b', linewidth=1)
            
                # ax.plot(range(start, end), [s[i+1][0] for s in self.state_history[start:end]], label=f'Agent {agent_labels[i]} Inventory',  color='navy',alpha=0.3, linewidth=2)
                # ax.plot(range(start, end), [s[i+1][1] for s in self.state_history[start:end]], label=f'Agent {agent_labels[i]} Backlog', color='b', linewidth=1)
                ax.legend(loc='upper left', fontsize='medium', markerscale=0.5)
                if i == 0:
                    ax.set_ylabel('Inventory & Backlog', fontsize='medium')
                ax.set_ylim([-1, max(self.factory_capacity, self.retailer_capacity) + 1])
            
            
            for i in range(2):
                ax = fig.add_subplot(4, 2, i + 5)
                ax.plot(range(start, end), [s[i*8] for s in self.state_history[start:end]], label=f'Agent {agent_labels[i]} Inventory', color='navy', alpha=0.3, linewidth=2)
                ax.plot(range(start, end), [s[i*8 + 2] for s in self.state_history[start:end]], label=f'Agent {agent_labels[i]} Stockout', color='b', linewidth=1)
                ax.legend(loc='upper left', fontsize='medium', markerscale=0.5)
                if i == 0:
                    ax.set_ylabel('Inventory & Stockout', fontsize='medium')
                ax.set_ylim([-1, max(self.factory_capacity, self.retailer_capacity) + 1])
                
            
             # Create a new subplot for the profit plot, spanning both columns (1 row, 1 column)
            ax = fig.add_subplot(4, 1, 4)
    
            # Plot the same profit data for both agents in one single plot
            mean_profit = sum(self.reward_history[start:end]) / (end - start)
            ax.plot(range(start, end), [r for r in self.reward_history[start:end]], label=f'Mean Profit: {mean_profit:.2f}', color="darkviolet", alpha=0.8, linewidth=0.65)
    
            # Customize labels and appearance
            ax.legend(loc='upper left', fontsize='medium', markerscale=0.5)
            ax.set_ylabel('Profit', fontsize='medium')
            ax.set_xlabel('Step')
    
            # Set y-limits based on the range of profit values
            ax.set_ylim([min(self.reward_history[start:end]) - 1, max(self.reward_history[start:end]) + 1])

            plt.tight_layout()
                
    
            # Save the plot
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                filename = os.path.join(log_dir, f'Step{ep_count}_ep{self.re_num}_V{self.v_f}T{self.task}_{current_date.strftime("%Y%m%d_%H%M%S")}.png')
                plt.savefig(filename, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()
        
        
        
        if ep_count % p == 0 and len(self.state_history) % p==0:
       
            start = max(0,len(self.state_history)-p)
         
            end = len(self.state_history)
         
  
            fig = plt.figure(figsize=(3 * 2, 6)) 
            colors = list(mcolors.TABLEAU_COLORS.keys())    
  
            for i in range(2):
                ax = fig.add_subplot(3, 2, i + 1)
                if i==0:
                    ax.plot(range(start , end),[s[5] for s in self.state_history[start:end]], label=f'Customer Demand',  color='b', alpha=0.85, linewidth=0.8)
                else:
                    ax.plot(range(start , end),[s[13] for s in self.state_history[start:end]], label=f'Retailer Order',  color='b', alpha=0.85, linewidth=0.8)
                #ax.plot(range(start , end),[s[i+1][5] for s in self.state_history[start:end]], label=f'Demand',  color='b', alpha=0.85, linewidth=0.8)
                ax.legend(loc='upper left', fontsize='medium', markerscale=0.5)
                if i == 0:
                    ax.set_ylabel('Demand', fontsize='medium')
                ax.set_ylim([-1, max(self.factory_capacity, self.retailer_capacity) + 1]) 
  
            for i in range(2):
                ax = fig.add_subplot(3, 2, i + 3)
                ax.plot(range(start, end), [a[i*3] for a in self.action_history[start:end]], label=f'{agent_labels[i]} Order', color='mediumslateblue',alpha=0.85, linewidth=0.8)
                ax.legend(loc='upper left', fontsize='medium', markerscale=0.5)
                if i == 0:
                    ax.set_ylabel('Order', fontsize='medium')
                ax.set_ylim([-1, max(self.factory_capacity, self.retailer_capacity) + 1]) 
                
            for i in range(2):
                ax = fig.add_subplot(3, 2, i + 5)
                ax.plot(range(start, end), [a[i*3+1] for a in self.action_history[start:end]], label=f'{agent_labels[i]} Price', color="lightsteelblue", alpha=0.4, linestyle='--',linewidth=1.7)
                ax.legend(loc='upper left', fontsize='medium', markerscale=0.5)
                if i == 0:
                    ax.set_ylabel('Price', fontsize='medium')
                ax.set_ylim([-1, max(self.factory_capacity, self.retailer_capacity) + 1]) 
        
            plt.tight_layout()
            
  
       
            if log_dir:
                
                os.makedirs(log_dir, exist_ok=True)
                filename = os.path.join(log_dir, f'Single_Step{ep_count}_ep{self.re_num}_V{self.v_f}T{self.task}_{current_date}.png')
                plt.savefig(filename, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()
    
        # Save the CSV after rendering
        filename = f"./POV{self.v_f}T{self.task}.csv"
        self.save_to_csv(filename)

 
  
class GymEnvironment(POSingAgent1W1F, gymnasium.Env):  
    def __init__(self, *args, **kwargs):  
        super().__init__(*args, **kwargs)  
        
        
        
class POSAgent1W1F_V1T1(POSingAgent1W1F, gymnasium.Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v_f = 1
        self.task = 1
       
             
class POSAgent1W1F_V1T2(POSingAgent1W1F, gymnasium.Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v_f = 1
        self.task = 2
 
class POSAgent1W1F_V1T3(POSingAgent1W1F, gymnasium.Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v_f = 1
        self.task = 3
        
        
class POSAgent1W1F_V1T4(POSingAgent1W1F, gymnasium.Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v_f = 1
        self.task = 4
        
        
class POSAgent1W1F_V2T1(POSingAgent1W1F, gymnasium.Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v_f = 2
        self.task = 1
        
class POSAgent1W1F_V2T2(POSingAgent1W1F, gymnasium.Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v_f = 2
        self.task = 2
        
class POSAgent1W1F_V2T3(POSingAgent1W1F, gymnasium.Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v_f = 2
        self.task = 3
  
  
class POSAgent1W1F_V3T1(POSingAgent1W1F, gymnasium.Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v_f = 3
        self.task = 1
        self.factory_capacity = 60
        self.retailer_capacity = 20 
        self.price_capacity = 60
        factory_capacity = self.factory_capacity 
        retailer_capacity = self.retailer_capacity 
        price_capacity = self.price_capacity
        self.action_space = MultiDiscrete([retailer_capacity ,price_capacity,self.v_f-1,retailer_capacity ,price_capacity,self.v_f-1])
        
class POSAgent1W1F_V3T2(POSingAgent1W1F, gymnasium.Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v_f = 3
        self.task = 2
        self.factory_capacity = 60
        self.retailer_capacity = 20 
        self.price_capacity = 60
        factory_capacity = self.factory_capacity 
        retailer_capacity = self.retailer_capacity 
        price_capacity = self.price_capacity
        self.action_space = MultiDiscrete([retailer_capacity,price_capacity,self.v_f-1,retailer_capacity ,price_capacity,self.v_f-1])
        
class POSAgent1W1F_V3T3(POSingAgent1W1F, gymnasium.Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v_f = 3
        self.task = 3
        self.factory_capacity = 60
        self.retailer_capacity = 20 
        self.price_capacity = 60
        factory_capacity = self.factory_capacity 
        retailer_capacity = self.retailer_capacity 
        price_capacity = self.price_capacity
        self.action_space = MultiDiscrete([retailer_capacity,price_capacity,self.v_f-1,retailer_capacity ,price_capacity,self.v_f-1])

 
 

if __name__ == '__main__':  

    env = POSingAgent1W1F()  
    episides = 10  
    for i in range(episides):   
        obs,_ = env.reset()  
        while True:  
            print('+++++++++++++++++egthrget5++++++++',i,'++++++++++++++++++++++++++')  

            obs, rew, done, _,info = env.step(   
                env.action_space.sample()   

            )   
            if done :  
                break  
    assert done  
    print('obs',obs)   
    print('rew',rew)   
    print('done',done)   
    print('info',info)   

 
 

   

 
 

     

 