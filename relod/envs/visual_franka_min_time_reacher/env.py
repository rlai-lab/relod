from relod.envs.visual_franka_dense_reacher.franka_env import FrankaPanda_Visual_Reacher_V0
import numpy as np
import time
import cv2

# TODO: size_tol

class FrankaPanda_Visual_Min_Reacher(FrankaPanda_Visual_Reacher_V0):
    def __init__(self, dt=0.04, 
    image_history_size=3, 
    image_width=160, 
    image_height=90, 
    episode_length=8, 
    camera_index=0, 
    seed=9, 
    size_tol=0.015):
        super(FrankaPanda_Visual_Min_Reacher, self).__init__(dt, image_history_size, image_width, image_height, episode_length, camera_index, seed)
        self._size_tol = size_tol
        self.seed(seed)

    def _get_mask(self, image_m):
        image = image_m.copy()
        image = image[:, :, -3:]
        lower = [0, 0, 120]
        upper = [120, 90, 255]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(image, lower, upper)

        hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red color
        low_red = np.array([151, 155, 84])
        high_red = np.array([179, 255, 255])
        mask = cv2.inRange(hsv_frame, low_red, high_red)
        kernel = np.ones((3, 3), 'uint8')
        mask = cv2.dilate(mask, kernel, iterations=5)
        mask = cv2.erode(mask, kernel, iterations=5)
        return mask

    def _compute_target_size(self, mask):  
        target_size = np.sum(mask/255.) / mask.size
        print(target_size)
        return target_size
    
    def step(self, action, pose_vel_limit=0.3):
        self.ep_time += self.dt
        self.robot_status.enable()
        
        # limit joint action
        action = action.reshape(-1)
        action = np.clip(action, -self.joint_action_limit, self.joint_action_limit)
        # convert joint velocities to pose velocities
        pose_action = np.matmul(self.get_robot_jacobian(), action)

        # limit action
        pose_action[:3] = np.clip(pose_action[:3], -pose_vel_limit, pose_vel_limit)

        # safety
        out_boundary = self.out_of_boundaries()
        pose_action[:3] = self.safe_actions(pose_action[:3])

        # calculate joint actions
        d_angle =  np.array(self.euler_from_quaternion(self.reset_ee_quaternion)) - np.array(self.euler_from_quaternion(self.ee_orientation))
        for i in range(3):
            if d_angle[i] < -np.pi:
                d_angle[i] += 2*np.pi
            elif d_angle[i] > np.pi:
                d_angle[i] -= 2*np.pi
        d_angle *= 0.5

        d_X = pose_action
        
        if out_boundary:
            d_X[3:] = 0
            action = self.get_joint_vel_from_pos_vel(d_X)

        action = self.handle_joint_angle_in_bound(action)
        self.apply_joint_vel(action)
        self.prev_action = action


        # pass time step duration

        done = False

        # Currently Episode end is handled in the task. 
        # TODO: Handle episode end
        # if self.ep_time >= (self.max_episode_duration-1e-3):
        #     done = True
        #     self.apply_joint_vel(np.zeros((7,)))
        #     info['TimeLimit.truncated'] = True
        
        delay = (self.ep_time + self.reset_time) - time.time()
        if delay > 0:
            time.sleep(np.float64(delay))

        # get next observation
        observation_robot = self.get_state()

        self.time_steps += 1
        
        # construct the state
        obs = np.concatenate((observation_robot["joints"], observation_robot["joint_vels"], action))
        
        mask = self._get_mask(observation_robot["image"])
        
        # print("cycte begin", self.tv)
        img = np.transpose(observation_robot["image"], (2, 1, 0))

        if self._image_history_size > 1:
            self._image_history[3:, :, :] = self._image_history[:-3, :, :]
            self._image_history[0:3, :, :] = img

        image = self._image_history.copy()
        prop = obs.copy()
        done = 0
        info = {}

        done = self._compute_target_size(mask) >= self._size_tol

        # if self._target_type == 'size':
        #     done = self._compute_target_size(image) >= self._size_tol
        # elif self._target_type == 'center':
        #     offset = self._compute_target_offset(image, (0, 0))
        #     done = offset[0] <= self._center_tol*2 and offset[1] <= self._center_tol*2
        # elif self._target_type == 'reward':
        #     r = self._compute_reward(image, prop)
        #     # print('r:',r)
        #     done = r >= self._reward_tol
        # elif self._target_type == 'size_center':
        #     offset = self._compute_target_offset(image, (0, 0))
        #     done = (self._compute_target_size(image) >= self._size_tol) and \
        #             (offset[0] <= self._center_tol*2 and offset[1] <= self._center_tol*2)
        # else:
        #     raise NotImplementedError()
        

        if done:
            self._reset = False
            self.apply_joint_vel(np.zeros((7,)))
            self.apply_joint_vel(np.zeros((7,)))
            # self.reset()

        reward = -1

        return image, prop, reward, done, info
    
    def seed(self, seed):
        super().seed(seed)
        np.random.seed(seed)

    
# def ranndom_policy_hits_vs_timeout():
#     total_steps = 1500  
#     input('go?')
#     steps_record = open(f"visual_franka_steps_record.txt", 'w')
#     hits_record = open(f"visual_franka_random_stat.txt", 'w')
#     dt = 0.04
#     for epi_len in [30]:
#         timeout = int(epi_len//dt)
#         env = FrankaPanda_Visual_Min_Reacher(episode_length=epi_len, camera_index=1, size_tol=0.050)
#         for seed in range(5):
#             np.random.seed(seed)
#             steps_record.write(f"epi_length={epi_len}s, seed={seed}: ")
#             # Experiment
#             hits = 0
#             steps = 0
#             epi_steps = 0

#             image, _ = env.reset()
            
#             while steps < total_steps:
#                 # TODO: Verify
#                 action = env.action_space.sample()

#                 # Receive reward and next state            
#                 _, _, _, done, _ = env.step(action)
                
#                 # print("Step: {}, Next Obs: {}, reward: {}, done: {}".format(steps, next_obs, reward, done))

#                 # Log
#                 steps += 1
#                 epi_steps += 1

#                 # Termination
#                 if done or epi_steps == timeout:
#                     #TODO: Verify
#                     t1 = time.time()
#                     env.reset()
#                     reset_step = (time.time() - t1) // dt
                    
#                     epi_steps = 0

#                     if done:
#                         hits += 1
#                     else:
#                         steps += int(reset_step)
                        
#                     steps_record.write(str(steps)+', ')

#             steps_record.write('\n')
#             hits_record.write(f"epi_length={epi_len}s, seed={seed}: {hits}\n")
#             env.reset()
#             env.close()

#     steps_record.close()
#     hits_record.close()






