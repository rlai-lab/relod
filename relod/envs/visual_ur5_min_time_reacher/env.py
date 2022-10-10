from envs.visual_ur5_reacher.reacher_env_min_time import ReacherEnv
import numpy as np
from senseact.utils import NormalizedEnv
import cv2, math
from statistics import mean
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

class MonitorTarget:
    def __init__(self):
        self.radius=7
        self.width=160
        self.height=90
        self.margin = 20
        mpl.rcParams['toolbar'] = 'None'
        plt.ion()
        self.fig = plt.figure()
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        self.fig.canvas.toolbar_visible = False
        self.ax = plt.axes(xlim=(0, self.width), ylim=(0, self.height))
        self.target = plt.Circle((0, 0), self.radius, color='red')
        self.ax.add_patch(self.target)
        plt.axis('off')

        figManager = plt.get_current_fig_manager()
        figManager.full_screen_toggle()

    def reset_plot(self):
        x, y = np.random.random(2)
        self.target.set_center(
            (self.radius + self.margin + x * (self.width - 2*self.radius - 2*self.margin),
             self.radius + self.margin + y * (self.height - 2*self.radius - 2*self.margin))
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.032)


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
                 target_type='size',
                 image_history=3,
                 joint_history=1,
                 episode_length=30,
                 dt=0.04,
                 size_tol=0.015,
                 center_tol=0.1,
                 reward_tol=1.0,
                ):
        self._image_width = image_width
        self._image_height = image_height
        self._dt = dt
        self._target_type = target_type
        self._size_tol = size_tol
        self._center_tol = center_tol
        self._reward_tol = reward_tol
        
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

    def _compute_reward(self, image, joint):
        """Computes reward at a given time step.
        Returns:
            A float reward.
        """
        image = np.transpose(image, [1,2,0])
        image = image[:, :, -3:]
        lower = [0, 0, 120]
        upper = [50, 50, 255]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(image, lower, upper)
        cv2.imshow('', mask)
        cv2.waitKey(1)
        
        size_x, size_y = mask.shape
        # reward for reaching task, may not be suitable for tracking
        if 255 in mask:
            xs, ys = np.where(mask == 255.)
            reward_x = 1 / 2  - np.abs(xs - int(size_x / 2)) / size_x
            reward_y = 1 / 2 - np.abs(ys - int(size_y / 2)) / size_y
            reward = np.sum(reward_x * reward_y) / self._image_width / self._image_height
        else:
            reward = 0
        reward *= 800
        reward = np.clip(reward, 0, 4)

        '''
        When the joint 4 is perpendicular to the mounting ground:
            joint 0 + joint 4 == 0
            joint 1 + joint 2 + joint 3 == -pi
        '''
        # chagne
        # scale = (np.abs(joint[0] + joint[4]) + np.abs(np.pi + np.sum(joint[1:4])))
        # return reward - scale
        return reward 

    @property
    def image_space(self):
        return self._env.observation_space['image']

    @property
    def proprioception_space(self):
        return self._env.observation_space['joint']

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):

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
        done = 0
        info = {}

        if self._target_type == 'size':
            done = self._compute_target_size(image) >= self._size_tol
        elif self._target_type == 'center':
            offset = self._compute_target_offset(image, (0, 0))
            done = offset[0] <= self._center_tol*2 and offset[1] <= self._center_tol*2
        elif self._target_type == 'reward':
            r = self._compute_reward(image, prop)
            # print('r:',r)
            done = r >= self._reward_tol
        elif self._target_type == 'size_center':
            offset = self._compute_target_offset(image, (0, 0))
            done = (self._compute_target_size(image) >= self._size_tol) and \
                    (offset[0] <= self._center_tol*2 and offset[1] <= self._center_tol*2)
        else:
            raise NotImplementedError()

        if done:
            self._reset = False
            self._env.stop_arm()

        reward = -1

        return image, prop, reward, done, info

    def close(self):
        self._env.close()
        
def random_policy_done_2_done_length():
    # Monitor
    mt = MonitorTarget()
    mt.reset_plot()
    
    input('go?')
    seed = 1
    np.random.seed(seed)
    total_dones = 50
    task = "visual ur5 reacher min time"
    episode_length_time = 30.0
    dt = 0.04
    size_tol = 0.015
    env = VisualReacherMinTimeEnv(seed=seed, episode_length=episode_length_time, dt=dt, size_tol=size_tol)

    # Experiment
    timeout = int(episode_length_time/dt)
    done_2_done_lens = []
    steps = 0
    while len(done_2_done_lens) < total_dones:
        env.reset()
        mt.reset_plot()
        epi_steps = 0
        done = 0
        done_2_done_steps = 0
        resets = 0
        while not done:
            action = env.action_space.sample()

            # step in the environment
            _, _, _, done, _ = env.step(action)

            # Log
            steps += 1
            epi_steps += 1
            done_2_done_steps += 1

            # Termination
            if epi_steps == timeout:
                resets += 1
                env.reset()
                mt.reset_plot()
                epi_steps = 0

        done_2_done_lens.append(done_2_done_steps)
        print('-'*50)
        print('Episode: {}, done_2_done steps: {},resets: {}, total steps: {}'.format(len(done_2_done_lens), done_2_done_steps, resets, steps))
        print('-'*50)

    with open(task+"_random_stat.txt", 'w') as out_file:
        for length in done_2_done_lens:
            out_file.write(str(length)+'\n')

        out_file.write(f"\nMean: {mean(done_2_done_lens)}")

if __name__ == '__main__':
    # np.random.seed(0)
    # env = VisualReacherMinTimeEnv(camera_id=0)
    # mt = MonitorTarget()
            
    # mt.reset_plot()
    # input('go?')
    # env.reset()
    # success, episodes = 0, 0
    # while episodes <= 20:
    #     action = env.action_space.sample()
        
    #     image, prop, reward, done, terminated, info = env.step(action)
        
    #     if done or terminated:
    #         episodes += 1
    #         env.reset()
    #         if done:
    #             success += 1
    #         print('episodes:', episodes)
    #         print('success:', success)
    random_policy_done_2_done_length()
    exit()
