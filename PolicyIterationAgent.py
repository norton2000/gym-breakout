import gym
import random
import numpy as np

def policy_evaluation(U, policy, env, gamma = 1):
    for i, state in enumerate(env.S):
        action = env.actionsName[policy[i]]
        nextState, reward, done = env.P[state][action]
        if not done:
            U[i] = reward + gamma * U[env.S.index(nextState)]
        else:
            U[i] = reward
    return U

def policyIteration(env):
    policy = np.full(env.stateNumber, 2)
    U = np.zeros(env.stateNumber)
    k = 0
    while True:
        print("starts iterations number: ", k+1, " of policyIteration")
        U = policy_evaluation(U, policy, env)
        unchanged = True
        for i, state in enumerate(env.S):
            best_action_index = policy[i]
            nextState, reward, done = env.P[state][env.actionsName[best_action_index]]
            action_compute_yet_index = best_action_index
            if not done:
                utilityMax = U[env.S.index(nextState)]
            else:
                utilityMax = reward
            for j, action in enumerate(env.actionsName):
                if j != action_compute_yet_index:
                    nextState, reward, done = env.P[state][action]
                    if not done:
                        state_selected_index = env.S.index(nextState)
                        if U[state_selected_index] > utilityMax:
                            best_action_index = j
                            utilityMax = U[state_selected_index]
                            unchanged = False
                    elif reward > utilityMax:
                        best_action_index = j
                        utilityMax = reward
                        unchanged = False
            policy[i] = best_action_index
            
        if unchanged:
            break
        k += 1
        if k==20:                                                                
            print("policyIteration terminated after ", k+1, " iterations")
            break
    return policy, U

env = gym.make('gym_breakout:breakout-v1')

policy, U = policyIteration(env)

state = env.initialState
done = False
print("Game Started!")
env.render()
while not done:
    i = policy[env.S.index(state)]
    action = 'left' if i==0 else 'right' if i==1 else 'none'
    print("Action selected: ", action)
    state,reward,done,info = env.step(action)
    env.render()
    if reward >= env.max_reward:
        print("YOU WIN!")
    print()