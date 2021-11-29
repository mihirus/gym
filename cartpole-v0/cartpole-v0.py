import gym 
import time
import numpy as np
import random
import matplotlib.pyplot as plt 

# State Machine flow: 
# 1) Start in state s 
# 2) Policy(a given s) is (1-epsilon)*max(Q(:, s)) + epsilon*min(Q(:,s))
#       which is simplified expression for (Sum over all a)(gamma*policy(a given s)*Q(a, s))
# 3) Land in state s'  
# 4) Update Q(a, s) = (1-alpha)*Q(a, s) + alpha*(r(s', a, s) + gamma*((1-epsilon)*max(Q(a', s')) - epsilon*Q(a, s)))
# 5) Repeat

# State info made available
# next_state - linear position, linear velocity, pole angle, pole velocity 

# env.step(action) returns 
      # next_state (object) - sensor measurements (pixels, acceleration) 
      # reward (float) - reward achieved by previous action 
      # done (boolean) - done? episode has terminated 
      # info (dict) - info for debugging, NOT for controlling agent

alpha = 0.2 # learning rate
gamma = 0.99 # discount
epsilon = 0 # exploration rate
num_episodes = 500
avg_time_chart = [0] # for plot

bins = 2 
q_table = np.zeros((bins**4, 2))
last_state = [0,0,0,0]
avg_time= [0,0,0,0,0,0,0,0,0,0]

state_ranges = np.zeros((4, 2))

delay_time = 0

# Gets moving average timesteps of last 10 episodes
def getAvgTime(): 
    return sum(avg_time)/len(avg_time)

# Gets the index of the bin in q_table from the current state
# Each state variable is assigned two bins, one high and one low
# The corresponding index in the q_table is a binary to decimal conversion
# Aka if bins of each state variable are 0, 1, 0, 1 respectively, q_table index is 5
def getStateInd(state):
    updateStateRanges(state)
    ind = 0
    other_ind = 0
    for state_ind in range(0, 4): 
        for bin_ind in range(0, bins):  
            current_state = state[state_ind] - state_ranges[state_ind, 0]
            current_state_allowed_range = state_ranges[state_ind, 1] - state_ranges[state_ind, 0]
            if(current_state <= (bin_ind + 1)*(current_state_allowed_range/bins)):
                ind += (bins**state_ind)*bin_ind
                break
    return ind

# Updates state ranges to contain largest observed negative and positive state values
def updateStateRanges(state): 
    for i in range(0, 4): 
        if(state[i] > state_ranges[i, 1]): 
            state_ranges[i, 1] = state[i]
        if(state[i] < state_ranges[i, 0]): 
            state_ranges[i, 0] = state[i]

# Updates Q table according to 
# Q(a, s) = (1-alpha)*Q(a, s) + alpha*(r(s', a, s) + gamma*((1-epsilon)*max(Q(a', s')) - epsilon*Q(a, s)))
def updateQ(last_state, action, next_state, reward): 
    if (action == 1): 
        action_q = 1
    else: 
        action_q = 0
    last_state_ind = getStateInd(last_state)    
    next_state_ind = getStateInd(next_state)    
    q_max = np.amax(q_table[next_state_ind,:])
    q_min = np.amin(q_table[next_state_ind,:])
    epsilon_adj = (200-getAvgTime())*epsilon
    q_table[last_state_ind, action_q] = (1-alpha)*q_table[last_state_ind, action_q] + alpha*(reward + gamma*((1-epsilon_adj)*q_max+epsilon_adj*q_min))

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env.reset()
    plt.axis('auto')
    plt.ion()
    plt.show()
    random.seed(version=2)
    for i_episode in range(num_episodes):
        next_state = env.reset() 
        # print(q_table)
        for j in range(1000):
            env.render() 
            X_rand = random.random()
            if (X_rand < epsilon and X_rand < epsilon/2): 
                action = 0 
            elif (X_rand < epsilon and X_rand >= epsilon/2): 
                action = 1
            else:
                if(q_table[getStateInd(last_state),0] > q_table[getStateInd(last_state), 1]): 
                    action = 0
                else: 
                    action = 1
            next_state, reward, done, info = env.step(action)
            if(done):
                if(j != 199):
                    reward = -100
                updateQ(last_state, action, next_state, reward) 
                avg_time.append(j)
                avg_time.pop(0)
                avg_time_chart.append(getAvgTime())
                plt.plot(avg_time_chart)
                plt.draw()
                plt.pause(0.001)
                print("Episode ", i_episode, " finished after", j, " timesteps, average of ", getAvgTime())
                break
            updateQ(last_state, action, next_state, reward) 
            last_state = next_state            
            time.sleep(delay_time)        
    env.close() 
