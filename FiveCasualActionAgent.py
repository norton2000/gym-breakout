import gym
import random

env = gym.make('gym_breakout:breakout-v1')
env.reset()

attempts = 5

actions = ["left", "right", "none"]
attempt = 1
done = False

for attempt in range(attempts):
    action = random.choice(actions)
    obs,reward,done,info = env.step(action)
    print("\n\nAction number: ", attempt+1)
    print("Action selected: ", action)
    print("\nObs state after the action:")
    print(obs)
    '''
    The obs is a Tuple (bar, (posB,dirB), bricks) where:
    bar = position of the bar (the bar is at (0,bar) in the game's grid)
    posB = a pair that indicate the position of the ball in the game's grid
    dirB = a pair that indicate the direction of the ball in the game's grid
    bricks = a List that represents where are the bricks at the final row. grid[i] = 1 if in the cell (rows-1,i) there is a wall, 0 if it is free
    '''
    env.render()
    if done:                        #Riuscirà a fare 5 azioni solo se non termina prima ovvero solo se le prime due mosse casuali sono 'right' e 'none' in qualsiasi ordine
        print("\nGame Over...")
        break
    
    
    
def randomAction():
    return random.choice(actions)