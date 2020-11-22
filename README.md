To install the gym environment do:
'pip install -e gym-breakout' in the directory where this repository is.

The code of the environment are in the file \gym_breakout\envs\breakout_env

There are two agents:

FiveCasualActionAgent.py that generates five casual actions and run this actions in the environment.
PolicyIterationAgent.py that implements and run a policy iteration algorithm to win the game.
