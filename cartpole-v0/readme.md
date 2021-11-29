# Cartpole using Q-learning

Solution to OpenAI Gym cartpole-v0, using Q-learning. Episodes end after 199 timesteps, and environment is considered solved when agent averages 195 timesteps before done, for 100 episodes. The strategy I used converges in 40-80 episodes.   

Reward is 1 no matter what, but I think they meant to provide 0 or -1 if done. There are 4 environment states (x_position, x_velocity, theta, and omega).  

### Design points & observations: 

Bad but somewhat workeable designs were usually unstable - their performance would quickly rise and fall. Several parts of the algorithm have a significant impact on stability, including # bins, implementation of bins, learning rate as a function of performance, exploration rate as a function of performance. 

I used 2 bins for each env states, making 16 states in Q-table. More bins per env state (like 3, 4) made training slower and more unstable, probably because many more Q-table state action pairs need to be explored. More bins also made movements more twitchy and responsive, for the same reason. My first "working" implementation of bins used a threshold of 0 for each env state. So if state 2 = -0.1, env state 2 would go into bin 0. This caused some instability in training, since each env state was not necessarily centered around 0. I reworked the binning to use the maximum observed range of each env state as the low and high extents of the bins, cutting up that range into n bins. 

I used a reward of -100 for episodes finished with less than 199 timesteps, aka a harsh punishment for failure. 










