from envs.visual_ur5_reacher.reacher_env_min_time import ReacherEnv
import numpy as np
from senseact.utils import NormalizedEnv
from remote_learner_ur5 import MonitorTarget
import cv2, math

def get_mask(image):
    image = np.transpose(image, [1,2,0])
    image = image[:,:,-3:]

    lower = [0, 0, 120]
    upper = [50, 50, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(image, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(mask, pts=contours, color=(255, 255, 255))
    
    return mask

def get_center(image):
    mask = get_mask(image)

    m = cv2.moments(mask)
    if math.isclose(m["m00"], 0.0, rel_tol=1e-6, abs_tol=0.0):
        x = 0
        y = 0
    else:
        x = int(m["m10"] / m["m00"])
        y = int(m["m01"] / m["m00"])

    cv2.circle(mask, (x, y), 1, (0,0,0), -1)
    cv2.imshow('mask', mask)
    cv2.waitKey(1)

    width = len(mask[0])
    height = len(mask)
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
                 target_type='center',
                 image_history=3,
                 joint_history=1,
                 episode_length=30,
                 dt=0.04,
                 tol=0.02
                ):
        self._tol = tol
        self._target_type = target_type
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
            target_type="reaching",
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

    def _compute_target_size(self, image):
        mask = get_mask(image)

        target_size = np.sum(mask/255.) / mask.size

        return target_size

    def _compute_target_offset(self, image, target_location):
        (x, y) = get_center(image)

        return abs(x-target_location[0]), abs(y-target_location[1])

    def _compute_reward(self, image):
        pass

    @property
    def image_space(self):
        return self._env._observation_space['image']

    @property
    def proprioception_space(self):
        return self._env._observation_space['joint']

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        assert not self._reset

        obs_dict = self._env.reset()
        image = obs_dict['image']
        prop = obs_dict['joint']

        self._reset = True

        return image, prop

    def step(self, action):
        assert self._reset
        obs_dict, reward, done, _ = self._env.step(action)
        image = obs_dict['image']
        prop = obs_dict['joint']
        terminated = done
        done = 0
        info = {}

        if self._target_type == 'size':
            done = self._compute_target_size(image) >= self._tol
        elif self._target_type == 'center':
            offset = self._compute_target_offset(image, (0, 0))
            done = offset[0] <= self._tol*2 and offset[1] <= self._tol*2
        elif self._target_type == 'reward':
            done = self._compute_reward(image) >= self._tol
        else:
            raise NotImplementedError()

        if done or terminated:
            self._reset = False

        reward = -1

        return image, prop, reward, done, terminated, info

if __name__ == '__main__':
    np.random.seed(0)
    env = VisualReacherMinTimeEnv(camera_id=2)
    mt = MonitorTarget()
    mt.reset_plot()
    mt.reset_plot()
    input('go?')
    env.reset()
    success, episodes = 0, 0
    while episodes <= 20:
        action = env.action_space.sample()
        image, prop, reward, done, terminated, info = env.step(action)
        
        if done or terminated:
            episodes += 1
            env.reset()
            if done:
                success += 1
            print('episodes:', episodes)
            print('success:', success)
