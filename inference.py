import numpy as np
import time
from handler import environment
from agent import DQNAgent

LOAD_MODEL = "models/2x256__-17350.00max_-22770.00avg_-27850.00min__1562545118.model"

env = environment()

if __name__ == '__main__':
    agent = DQNAgent()

for round in range(5):
    # start the environment
    current_state = env.reset()

    done = False

    while not done:

        last_time = time.time()
       
        action = np.argmax(agent.get_qs(current_state))

        new_state, reward, done = env.step(action)

        current_state = new_state
        
        print(f"Action: {action} | Loop took: {time.time() - last_time}")

    print(f'Final Score for round{round}: {env.score}')