import numpy as np
import time
from handler import environment
from agent import DQNAgent

LOAD_MODEL = "models/2x256__-17350.00max_-22770.00avg_-27850.00min__1562545118.model"

ROUNDS = 5

env = environment()

if __name__ == '__main__':
    agent = DQNAgent()

    for round in range(ROUNDS):
        # start the environment
        current_state = env.reset()

        done = False

        while not done:

            last_time = time.time()
        
            # determine action with highest Q-value
            action = np.argmax(agent.get_qs(current_state))
            # apply action and receive new state
            new_state, reward, done = env.step(action)
            # update state
            current_state = new_state
            
            print(f"Action: {action} | Loop took: {time.time() - last_time}")

        print(f'Final Score for round{round}: {env.score}')
        time.sleep(1)