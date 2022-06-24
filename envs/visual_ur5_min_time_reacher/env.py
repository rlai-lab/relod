from envs.visual_ur5_reacher.reacher_env import ReacherEnv
import numpy as np
from senseact.utils import NormalizedEnv
from remote_learner_ur5 import MonitorTarget
from gym.spaces import Box
import cv2
import math

def get_center(image):
    image = np.transpose(image, [1, 2, 0])[:,:,-3:]
    image = np.array(image)
    lower = [120, 0, 0]
    upper = [255, 50, 50]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(image, lower, upper)

    m = cv2.moments(mask)
    if math.isclose(m["m00"], 0.0, rel_tol=1e-6, abs_tol=0.0):
        x = 0
        y = 0
    else:
        x = int(m["m10"] / m["m00"])
        y = int(m["m01"] / m["m00"])
        
    cv2.circle(image, (x, y), 1, (255, 255, 255), -1)
    cv2.imshow('', image)
    cv2.waitKey(1)
    width = len(image[0])
    height = len(image)
    x = -1.0 + x/width*2
    y = -1.0 + y/height*2
    return x, y

class VisualReacherMinTimeEnv:
    def __init__(self,
                 setup='Visual-UR5-min-time',
                 ip='129.128.159.210',
                 seed=9,
                 camera_id=0,
                 image_width=160,
                 image_height=90,
                 target_type='reaching',
                 image_history=3,
                 joint_history=1,
                 episode_length=30,
                 dt=0.04,
                ):
        
        # state
        np.random.seed(seed)
        rand_state = np.random.get_state()
        env = ReacherEnv(
            setup=setup,
            host=ip,
            dof=5,
            camera_id=camera_id,
            image_width=image_width,
            image_height=image_height,
            channel_first=True,
            control_type="velocity",
            target_type=target_type,
            image_history=image_history,
            joint_history=joint_history,
            reset_type="zero",
            reward_type="dense",
            derivative_type="none",
            deriv_action_max=5,
            first_deriv_max=2,
            accel_max=1.4,
            speed_max=2,
            speedj_a=1.4,
            episode_length_time=episode_length,
            episode_length_step=None,
            actuation_sync_period=1,
            dt=dt,
            run_mode="multiprocess",
            rllab_box=False,
            movej_t=2.0,
            delay=0.0,
            random_state=rand_state
        )

        self._env = NormalizedEnv(env)
        env.start()

        self._reset = False

    @property
    def image_space(self):
        # change
        image_shape = (0, 0, 0)
        return Box(low=0, high=255, shape=image_shape)
        #return self._env._observation_space['image']

    @property
    def proprioception_space(self):
        high = self._env._observation_space['joint'].high
        high = np.append(high, [1, 1])
        low = self._env._observation_space['joint'].low
        low = np.append(low, [-1, -1])
        return Box(low=low, high=high)
        #return self._env._observation_space['joint']

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        assert not self._reset

        obs_dict = self._env.reset()
        image = obs_dict['image']
        x, y = get_center(image)
        prop = obs_dict['joint']
        prop = np.append(prop, [x, y])

        self._reset = True

        return None, prop

    def step(self, action):
        assert self._reset
        obs_dict, reward, done, _ = self._env.step(action)
        image = obs_dict['image']
        x, y = get_center(image)
        prop = obs_dict['joint']
        prop = np.append(prop, [x, y])
        terminated = done
        done = 0
        info = {}
        info['reward'] = reward
        if reward >= 1:
            done = 1

        if done or terminated:
            self._reset = False

        reward = -1

        return None, prop, reward, done, terminated, info

if __name__ == '__main__':
    np.random.seed(9)
    env = VisualReacherMinTimeEnv()
    mt = MonitorTarget()
    mt.reset_plot()
    mt.reset_plot()
    mt.reset_plot()
    mt.reset_plot()
    env.reset()
    success, episodes = 0, 0
    while episodes <= 20:
        action = env.action_space.sample()
        image, prop, reward, done, terminated, info = env.step(action)
        print('reward:', info['reward'])
        
        if done or terminated:
            episodes += 1
            env.reset()
            if done:
                success += 1
            print('episodes:', episodes)
            print('success:', success)
