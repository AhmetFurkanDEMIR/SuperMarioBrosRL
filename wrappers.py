# @author: Ahmet Furkan DEMIR

import gym
import gym_super_mario_bros
from configPy import getEnv
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from ray.rllib.env.atari_wrappers import (MonitorEnv,
                                          NoopResetEnv,
                                          WarpFrame,
                                          FrameStack)


class EpisodicLifeEnv(gym.Wrapper):
    
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game
        over. Done by DeepMind for the DQN and co. since it helps value
        estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped._life
        if self.lives > lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few fr
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped._life
        return obs


class CustomReward(gym.Wrapper):

    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self._current_score = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        # info: {'coins': 0, 'flag_get': False, 'life': 2, 'score': 0, 'stage': 1, 'status': 'small', 'time': 393, 'world': 1, 'x_pos': 244, 'y_pos': 102}

        """
        
        v: the difference in agent x values between states
        in this case this is instantaneous velocity for the given step
        v = x1 - x0
            x0 is the x position before the step
            x1 is the x position after the step
        moving right ⇔ v > 0
        moving left ⇔ v < 0
        not moving ⇔ v = 0

        c: the difference in the game clock between frames
        the penalty prevents the agent from standing still
        c = c0 - c1
            c0 is the clock reading before the step
            c1 is the clock reading after the step
        no clock tick ⇔ c = 0
        clock tick ⇔ c < 0

        d: a death penalty that penalizes the agent for dying in a state
        this penalty encourages the agent to avoid death
        alive ⇔ d = 0
        dead ⇔ d = -15

        r = v + c + d
        The reward is clipped into the range (-15, 15).

        """

        if reward > 0:
            reward *= 10

        else:

            reward -= 1
            reward *= 10

        reward += (info["score"] - self._current_score)/10.
        self._current_score = info["score"]

        if info["flag_get"]:
            reward += 500

        return state, reward, done, info

def wrap_mario():

    env = gym_super_mario_bros.make(getEnv())
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MonitorEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = EpisodicLifeEnv(env)
    env = WarpFrame(env,84)
    env = CustomReward(env)
    env = FrameStack(env, 4)
    return env