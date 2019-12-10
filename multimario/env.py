## multi-obejcetive super mario bros
## modified by Runzhe Yang on Dec. 8, 2018

import gym
import os
import random
from itertools import chain

import numpy as np

from torch.multiprocessing import Pipe, Process
import cv2
from collections import deque

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


class MoMarioEnv(Process):
    def __init__(
            self,
            args,
            env_idx,
            child_conn,
            history_size=4,
            h=84,
            w=84):
        super(MoMarioEnv, self).__init__()
        self.daemon = True
        self.env = JoypadSpace(
            gym_super_mario_bros.make(args.env_id), SIMPLE_MOVEMENT)

        self.is_render = args.render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.coin = 0
        self.x_pos = 0
        self.time = 0
        self.score = 0
        self.n_mo = 5
        self.morall = np.zeros(self.n_mo)
        self.recent_rlist = deque(maxlen=100)
        self.recent_morlist = deque(maxlen=100)
        self.child_conn = child_conn
        self.life_done = args.life_done
        self.single_stage = args.single_stage
        self.stage_bonus = 0

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.reset()

    def run(self):
        super(MoMarioEnv, self).run()
        while True:
            action = self.child_conn.recv()
            if self.is_render:
                self.env.render()
            obs, reward, done, info = self.env.step(action)

            if self.single_stage and info["flag_get"]:
                self.stage_bonus = 10000
                done = True

            ''' Construct Multi-Objective Reward'''#####################################
            # [x_pos, time, death, coin]
            moreward = []
            # 1. x position
            xpos_r = info["x_pos"] - self.x_pos
            self.x_pos = info["x_pos"]
            # resolve an issue where after death the x position resets
            if xpos_r < -5:
                xpos_r = 0
            moreward.append(xpos_r)
            
            # 2. time penaltiy 
            time_r = info["time"] - self.time
            self.time = info["time"]
            # time is aways decreasing
            if time_r > 0:
                time_r = 0
            moreward.append(time_r)

            # 3. death 
            if self.lives > info['life']:
                death_r = -25
            else:
                death_r = 0
            moreward.append(death_r)

            # 4. coin
            coin_r = (info['coins'] - self.coin) * 100
            self.coin = info['coins']
            moreward.append(coin_r)

            # 5. enemy
            enemy_r = info['score'] - self.score
            if coin_r > 0 or done:
                enemy_r = 0
            self.score = info['score']
            moreward.append(enemy_r)

            ############################################################################
            

            if self.life_done:
                # when Mario loses life, changes the state to the terminal
                # state.
                if self.lives > info['life'] and info['life'] > 0:
                    force_done = True
                    self.lives = info['life']
                else:
                    force_done = done
                    self.lives = info['life']
            else:
                # normal terminal state
                force_done = done

            # reward range -15 ~ 15
            r = reward / 15
            self.rall += reward

            self.morall += np.array(moreward)
            mor = np.array(moreward) * self.n_mo / 15

            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = self.pre_proc(obs)

            self.steps += 1

            score = info['score']+self.stage_bonus

            if done:
                self.recent_rlist.append(self.rall)
                self.recent_morlist.append(self.morall)
                print(
                    "[Episode {}({})]\tStep: {}\tScore: {}\tMoReward: {}\tRecent MoReward: {}\tcoin: {}\tcurrent x:{}".format(
                        self.episode,
                        self.env_idx,
                        self.steps,
                        score,
                        self.morall,
                        np.mean(
                            self.recent_morlist, axis=0),
                        info['coins'],
                        info['x_pos']))

                self.history = self.reset()

            self.child_conn.send(
                [self.history[:, :, :], r, force_done, done, mor, score])

    def reset(self):
        self.steps = 0
        self.episode += 1
        self.rall = 0
        self.lives = 3
        self.coin = 0
        self.x_pos = 0
        self.time = 0
        self.score = 0
        self.stage_bonus = 0
        self.morall = np.zeros(self.n_mo)
        self.get_init_state(self.env.reset())
        return self.history[:, :, :]

    def pre_proc(self, X):
        # grayscaling
        x = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
        # resize
        x = cv2.resize(x, (self.h, self.w))
        x = np.float32(x) * (1.0 / 255.0)

        return x

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)
