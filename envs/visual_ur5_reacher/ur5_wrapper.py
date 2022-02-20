import time
import utils
import argparse
import gym

import numpy as np

from envs.visual_ur5_reacher.reacher_env import ReacherEnv
from senseact.utils import tf_set_seeds, NormalizedEnv


def make_env(setup='Visual_UR5',
             ip='129.128.159.210',
             seed=9,
             camera_id=0,
             image_width=160,
             image_height=120,
             target_type='stationary',
             image_history=3,
             joint_history=1,
             episode_length=4.0,
             dt=0.04):
    # state
    np.random.seed(seed)
    rand_state = np.random.get_state()
    # Create Visual UR5 Reacher environment
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
            speed_max=0.3,
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
    env = NormalizedEnv(env)
    env.start()
    return env

class UR5Wrapper():
    def __init__(self,
                 setup='Visual_UR5',
                 ip='129.128.159.210',
                 seed=9,
                 camera_id=0,
                 image_width=160,
                 image_height=120,
                 target_type='stationary',
                 image_history=3,
                 joint_history=1,
                 episode_length=4.0,
                 dt=0.04,
                 ignore_joint=False,
                 ):
        self.env = make_env(
                        setup,
                        ip,
                        seed,
                        camera_id,
                        image_width,
                        image_height,
                        target_type,
                        image_history,
                        joint_history,
                        episode_length,
                        dt,
                        )

        self.observation_space = self.env.observation_space['image']
        self.ignore_joint = ignore_joint
        if ignore_joint:
            self.state_space = gym.spaces.Box(low=0, high=1., shape=(0, ), dtype=np.float32)
            pass
        else:
            self.state_space = self.env.observation_space['joint']

        self.action_space = self.env.action_space

    def step(self, action):
        obs_dict, reward, done, _ = self.env.step(action)
        if self.ignore_joint:
            return obs_dict['image'], None, reward, done, _
        else:
            return obs_dict['image'], obs_dict['joint'], reward, done, _

    def reset(self):
        obs_dict = self.env.reset()
        if self.ignore_joint:
            return obs_dict['image'], None
        else:
            return obs_dict['image'], obs_dict['joint']

    def terminate(self):
        self.env.terminate()


def interaction_random_pi():
    n_episodes = 10
    env = UR5Wrapper(setup='Visual-UR5',
                     ip='129.128.159.210',
                     seed=9,
                     camera_id=0,
                     image_width=160,
                     image_height=120,
                     target_type='reaching',
                     image_history=3,
                     joint_history=1,
                     episode_length=4.0,
                     dt=0.04,
                     ignore_joint=False)

    for i_episode in range(n_episodes):
        img, prop = env.reset()
        done = False
        tic = time.time()

        while not done:
            action = np.random.uniform(low=-1, high=1, size=5)
            next_img, next_prop, reward, done, _ = env.step(action)
            img = next_img
            prop = next_prop
        print("Episode {} took {}s".format(i_episode, time.time() - tic))

def interaction_nn():
    from algo.sac_rad_agent import SACRADPerformer
    from task_ur5_visual_reacher import config

    parser = argparse.ArgumentParser(description='Local remote visual UR5 Reacher')
    # environment
    parser.add_argument('--setup', default='Visual-UR5')
    parser.add_argument('--env_name', default='Visual-UR5', type=str)
    parser.add_argument('--ur5_ip', default='129.128.159.210', type=str)
    parser.add_argument('--camera_id', default=0, type=int)
    parser.add_argument('--image_width', default=160, type=int)
    parser.add_argument('--image_height', default=90, type=int)
    parser.add_argument('--target_type', default='reaching', type=str)
    parser.add_argument('--random_action_repeat', default=1, type=int)
    parser.add_argument('--agent_action_repeat', default=1, type=int)
    parser.add_argument('--image_history', default=3, type=int)
    parser.add_argument('--joint_history', default=1, type=int)
    parser.add_argument('--ignore_joint', default=False, action='store_true')
    parser.add_argument('--episode_length_time', default=4.0, type=float)
    parser.add_argument('--dt', default=0.04, type=float)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float)
    # train
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--env_steps', default=100000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--async_mode', default=True, action='store_true')
    parser.add_argument('--max_updates_per_step', default=2, type=float)
    parser.add_argument('--update_every', default=50, type=int)
    parser.add_argument('--update_epochs', default=50, type=int)
    # critic
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    parser.add_argument('--bootstrap_terminal', default=1, type=int)
    # actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    # agent
    # parser.add_argument('--remote_ip', default='localhost', type=str)
    parser.add_argument('--remote_ip', default='192.168.0.105', type=str)
    parser.add_argument('--port', default=9876, type=int)
    parser.add_argument('--mode', default='ro', type=str, help="Modes in ['r', 'o', 'ro'] ")
    # misc
    parser.add_argument('--args_port', default=9630, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=True, action='store_true')
    # parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_model_freq', default=10000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--lock', default=False, action='store_true')

    args = parser.parse_args()
    utils.set_seed_everywhere(args.seed, None)

    env = UR5Wrapper(
        setup = args.setup,
        ip = args.ur5_ip,
        seed = args.seed,
        camera_id = args.camera_id,
        image_width = args.image_width,
        image_height = args.image_height,
        target_type = args.target_type,
        image_history = args.image_history,
        joint_history = args.joint_history,
        episode_length = args.episode_length_time,
        dt = args.dt,
        ignore_joint = args.ignore_joint,
    )

    args.image_shape = env.observation_space.shape
    args.proprioception_shape = env.state_space.shape
    args.action_shape = env.action_space.shape
    args.net_params = config
    args.env_action_space = env.action_space

    performer = SACRADPerformer(args)
    # First inference call takes a lot of time (~1 min). Do it before agent-env interaction loop
    img, prop = env.reset()
    action = performer.sample_action((img, prop), args.init_steps + 1)

    n_episodes = 10
    for i_episode in range(n_episodes):
        img, prop = env.reset()
        done = False
        tic = time.time()
        step = args.init_steps + 1
        while not done:
            action = performer.sample_action((img, prop), step)
            next_img, next_prop, reward, done, _ = env.step(action)
            img = next_img
            prop = next_prop
            step += 1
        print("Episode {} took {}s".format(i_episode, time.time() - tic))

if __name__ == '__main__':
    interaction_random_pi()
