import cv2
import numpy as np
from windowGrab import windowGrab
from memory_helper import get_pid, calculate_final_address, read_value_from_memory
from keras.callbacks import TensorBoard
import tensorflow as tf
from key_helper import UP, DOWN, LEFT, RIGHT, NOTHING, SPACE, ESC, press
import traceback
import time

class environment():
    DEATH_REWARD = -50_000
    IDLE_REWARD = -100
    SCORE_REWARD = 3_000
    TEMP_SCORE_REWARD = 200
    BIG_SCORE_REWARD = 6_000

    ACTION_SPACE_SIZE = 5

    WIDTH = 100
    HEIGHT = 75

    OBSERVATION_SPACE_VALUES = (HEIGHT, WIDTH, 3)

    def __init__(self):
        self.new_start = True
        self.pid = get_pid('superflight.exe')
        self.score_addr, self.phandler = calculate_final_address(self.pid, [0xA0, 0x2F0, 0x20, 0x9C])
        self.temp_score_addr, _ = calculate_final_address(self.pid, [0xA0, 0x2F0, 0x20, 0xA0])
        self.game_state_addr, _ = calculate_final_address(self.pid, [0xA0, 0x2F0, 0x20, 0x90])
        self.score = 0
        self.temp_score = 0
        self.old_score = 0
        self.episode_step = 0

    def reset(self):
        self.score = 0
        self.temp_score = 0
        self.old_score = 0
        self.game_state = 0
        self.episode_step = 0

        # start the next game
        press(SPACE)

        # return inintal observation
        screen = windowGrab("SUPERFLIGHT", self.new_start)
        screen = cv2.resize(screen, (self.WIDTH,self.HEIGHT))

        return screen

    def apply_action(self, action):
        # this is supposed to prevent the agent from accidentaly fiddling in the menus
        in_menu = False
        if self.game_state == 1 or self.game_state == 2:
            in_menu = True

        if action == 0 and not in_menu:
            press(UP)
        elif action == 1 and not in_menu:
            press(DOWN)
        elif action == 2 and not in_menu:
            press(LEFT)
        elif action == 3 and not in_menu:
            press(RIGHT)
        elif action == 4 and not in_menu:
            press(NOTHING)
        elif action == 5 and not in_menu:
            press(SPACE)
        else:
            press(ESC)



    def step(self, action):
        try:
            #last_time = time.time()

            self.episode_step += 1

            # take action chosen by agent
            self.apply_action(action)

            # grab window screen
            screen = windowGrab("SUPERFLIGHT", self.new_start)
            new_observation = cv2.resize(screen, (self.WIDTH,self.HEIGHT))

            if self.new_start:
                self.new_start = False

            self.old_score = self.score
            self.score = read_value_from_memory(self.score_addr, self.phandler)
            self.temp_score = read_value_from_memory(self.temp_score_addr, self.phandler)
            self.game_state = read_value_from_memory(self.game_state_addr, self.phandler)

            done = False

            # game_state: 0=playing; 1=paused; 2=crashed 
            if self.game_state == 1:
                self.apply_action(5)
            # if agent crashes, the epsiode is over
            elif self.game_state == 2:
                done = True
                self.apply_action(5)
                

            # reward for this step:
            
            # agent died
            if done:
                reward = self.DEATH_REWARD

            # agent is collecting points
            elif self.temp_score != 0:
                reward = self.TEMP_SCORE_REWARD

            # agent finished a combo and got the points
            elif self.score > self.old_score:

                # agent did a good combo and scored big
                if self.score - self.old_score > 200:
                    reward = self.BIG_SCORE_REWARD

                # agent got a small amount of points
                else:
                    reward = self.SCORE_REWARD

            # agnet is just idling in the air
            else:
                reward = self.IDLE_REWARD

            #print(self.temp_score, self.score, time.time()-last_time)

            return new_observation, reward, done
    
        except Exception as e:
            print("some problem during handler main loop", e)
            print(traceback.format_exc())
            self.new_start = True

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
