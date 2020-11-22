from gym.envs.registration import register

register(
    id='breakout-v1',
    entry_point='gym_breakout.envs:breakoutEnv',
)
