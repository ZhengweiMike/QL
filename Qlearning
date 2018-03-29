import numpy as np
import random as rand
import scipy.optimize as spo
import pandas as pd

class QLearner(object):

    def __init__(self,num_states,num_actions, alpha , gamma, ebsilon, decay,cov):
        
        '''
        states: 500 stocks marked as 1..500
        alpha: learning rate
        gamma: discount rate
        ebislon: random action probability
        decay: random p decay rate
        actions: 499 stocks to choose to add in portfolio (Exclude those included in portfolio)
        
        
        walking only one route with no going back per Q table
        No need to update Q table
        
        at Current State Si, i stocks included in portfolio
        
        State[i] = min wTQw s.t Sum w = 1, w is ix1
        Q[Si,a] = State[i] - min w'TQ'w'  s.t Sum w' = 1, w' is (i+1) x 1
        Q[Si+1,a] = State[i] - min w''T Q'' w'' s.t Sum w'' = 1, w' is (i+1)x2 (w' include the previous action) 
        '''
        # Init Variables and Q-table
        self.num_actions = num_actions
        self.portfolio = []
        self.alpha = alpha
        self.gamma = gamma
        self.rar = ebsilon
        self.radr = decay
        self.cov = cov
        
        self.start = rand.randint(0,499)
        self.portfolio.append(self.start)
        

    def state_value(self, s):
        """
        s : portfolio compositions
        Q : partial Covariance Matrix
        """
        s = sorted(s)
        equal_weight = 1.0/len(s)
        w = np.asarray([equal_weight]*len(s))
        cons = ({'type':'eq','fun': lambda w: np.sum(w)-1.0})
        bound = np.asarray([(0.0,1.0)]*len(s))
        Q = self.cov.loc[s,s]
        
        def f(w):
            return w.T.dot(Q).dot(w)
        
        value = spo.minimize(f,w,bounds=bound,constraints=cons).fun
        
        return value
        

    def query(self):
        
        """
        @summary: Select Next Action Based Variance Reward and future Variance Reward
                  Update The Portfolio by choosing different actions
        @returns: The selected action
        """ 
        
        
        x = set(range(0,500))
        
        if rand.uniform(0.0,1.0) <= self.rar:
            action = rand.randint(0,499)
            while action in self.portfolio:
                action = rand.randint(0,499)
        else:
            node_visited = []
            reward = []
            current_state = self.state_value(self.portfolio)
            for i in x - set(self.portfolio):
                temp = list(self.portfolio)
                temp.append(i)
                temp_value = self.state_value(temp)
                immediate_reward = current_state - temp_value
                future_reward = -10000
                for j in x - set(temp):
                    ahead_temp = list(temp)
                    ahead_temp.append(j)
                    ahead_value = self.state_value(ahead_temp)
                    if current_state - ahead_value > future_reward:
                        future_reward = current_state - ahead_value
                reward.append(immediate_reward + self.gamma * future_reward)
                node_visited.append(i)
            print(node_visited)
            print(reward)
            action = node_visited[np.argmax(reward)]
            print("Action Taken ,",action)
        # Take the action and update the portfolio            
        self.portfolio.append(action)
        # Update the random rate
        self.rar = self.rar * self.radr
        
        return action

    def run(self):
        
        while len(self.portfolio) < 30:
            print("1 is Done and Current portfolio is," , self.portfolio)
            self.query()
            
        return self.state_value(self.portfolio)


if __name__ == "__main__":

    cov = pd.read_excel("covariance.xlsx").values
    cov = pd.DataFrame(data=cov)
    num_states = 500
    num_actions = num_states - 1 # Completed Graph
    alpha = 0.2
    gamma = 0.6
    ebsilon = 0.3
    decay = 0.8
    ql = QLearner(num_states,num_actions,alpha,gamma,ebsilon,decay,cov)
    result = ql.run()
