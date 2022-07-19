from envs.visual_ur5_reacher.reacher_env_min_time import ReacherEnv
import numpy as np
from senseact.utils import NormalizedEnv
from remote_learner_ur5 import MonitorTarget
from gym.spaces import Box

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
                 tol=0.02
                ):
        self._tol = tol
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
        info['reward'] = reward
        if reward >= self._tol:
            done = 1

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
