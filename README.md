# Superflight-DQN

Superflight-DQN is a complete package with all the code needed to get started with training an AI agent for the game Superflight.

//GIF PLAYING

This agent has been trained for approximately 20K iterations of the game.

## SDQN provides:
* A screen capture as input state
* The current in-game score as integer value
* Events that can be given rewards/punishments
* A basic Deep-Q-Network that can be customized

## The Environment
The environment is an intermediary that provides the agent with information about the game state, takes in the agent's action and passes it to the game. This cycle is repeated at every time step.

The current state of the game is represented by what it displays. The entire game screen is captured and resized to be used by the agent. Additionally the reward (or punishment) for the previous action is calculated by checking for several events like idling, executing a combo, fininshing a combo with many or few points and player death. The agent updates its policies (the weigths of the network) according to the reward and the previous action and calculates the next action based on the current observation (state of the game).




