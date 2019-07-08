import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
from handler import *

LOAD_MODEL = "models/2x256__-21750.00max_-23690.00avg_-26250.00min__1562564159.model"

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 2_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 400  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -30_000  # For model save
SAVE_EVERY = 150
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 6_000

LEARNING_RATE = 0.001

# Exploration settings
epsilon = 0.35 # not a constant, going to be decayed
EPSILON_DECAY = 0.99975 # 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 5  # episodes
SHOW_PREVIEW = False

env = environment()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1337)
np.random.seed(1337)
tf.set_random_seed(1337)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


class DQNAgent:
    def __init__(self):
        #main model. This gets trained every step
        self.model=self.create_model()

        #Target model. This is what we .predict every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory=deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

    def create_model(self):
        if LOAD_MODEL is not None:
            print(f"Loading {LOAD_MODEL}")
            model = load_model(LOAD_MODEL)
            print(f"Model {LOAD_MODEL} loaded!")
            pass
        else:
            model = Sequential()

            model.add(Conv2D(256, (3,3), input_shape=(env.OBSERVATION_SPACE_VALUES)))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(2,2))
            model.add(Dropout(0.2))

            model.add(Conv2D(256, (3,3)))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(2,2))
            model.add(Dropout(0.2))

            model.add(Flatten())
            model.add(Dense(64))

            model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear"))
            model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X)/255, np.array(y), batch_size = MINIBATCH_SIZE, verbose = 0, shuffle = False, callbacks=[self.tensorboard] if terminal_state else None)

        if terminal_state:
            # updating to determine if we want to update target_model yet
            self.target_update_counter += 1

            
            if self.target_update_counter > UPDATE_TARGET_EVERY:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0

if __name__ == '__main__':
    agent = DQNAgent()

    for episode in tqdm(range(1, EPISODES+1), ascii=True, unit = "episode"):
        agent.tensorboard.step = episode

        episode_reward = 0
        step = 1
        current_state = env.reset()

        done = False

        while not done:

            last_time = time.time()

            # if env.game_state == 2:
            #     env.apply_action(5)
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state))
            else:
                action = np.random.randint(0, env.ACTION_SPACE_SIZE)

            # print(action)
            # if action == 4:
            #     print("-------------------------------------")

            new_state, reward, done = env.step(action)

            reward = reward if not done else env.DEATH_REWARD

            episode_reward += reward

            #save the state and action pairs
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            
            current_state = new_state
            
            print(f"Action: {action} | Loop took: {time.time() - last_time}")

        agent.train(done)
            
        step += 1

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if episode % SAVE_EVERY == 0 or EPISODES == episode:
                agent.model.save(f'models/{MODEL_NAME}__epsilon_{round(epsilon, 2)}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
            
        # let epsilon start over at the half way point
        # if episode % EPISODES/2 == 0:
        #     epsilon = 0.8
