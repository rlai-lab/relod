# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import time
import gym
import logging
import numpy as np
import senseact.devices.create2.create2_config as create2_config
from senseact import utils

from multiprocessing import Array, Value

from senseact.rtrl_base_env import RTRLBaseEnv
from senseact.devices.create2.create2_communicator import Create2Communicator
from senseact.envs.create2.create2_observation import Create2ObservationFactory
from senseact.sharedbuffer import SharedBuffer
from envs.create2_visual_docker.depstech_camera_communicator import CameraCommunicator
import cv2

class Create2VisualDockerEnv(RTRLBaseEnv, gym.Env):
    """Create2 environment for training it drive forward.
    By default this environment observes the infrared character (binary state for detecting the dock),
    bump sensor, and last action.
    The reward is the forward reward.  Episode ends early when the bump sensor is triggered.
    TODO:
        * move all common methods between this class and docking_env to a base class
    """

    def __init__(self, episode_length_time=30, port='/dev/ttyUSB0', obs_history=1, dt=0.015, image_shape=(0, 0, 0),
                 camera_id=0, **kwargs):
        """Constructor of the environment.
        Args:
            episode_length_time: A float duration of an episode defined in seconds
            port:                the serial port to the Create2 (eg. '/dev/ttyUSB0')
            obs_history:         the number of observation history to keep
            dt:                  the cycle time in seconds
            auto_unwind:         boolean of whether we want to execute the auto cable-unwind code
            rllab_box:           whether we are using rllab algorithm or not
            **kwargs:            the remaining arguments passed to the base class
        """
        self._obs_history = obs_history
        self._episode_step_ = Value('i', 0)
        self._episode_length_time = episode_length_time
        self._episode_length_step = int(episode_length_time / dt)
        self._internal_timing = 0.015
        self._hsv_mask = ((60, 0, 0), (80, 255, 255))
        self._min_battery = 1200
        self._max_battery = 1850

        # get the opcode for our main action (only 1 action)
        self._main_op = 'drive_direct'
        self._extra_ops = ['safe', 'seek_dock', 'drive']
        main_opcode = create2_config.OPCODE_NAME_TO_CODE[self._main_op]
        extra_opcodes = [create2_config.OPCODE_NAME_TO_CODE[op] for op in self._extra_ops]

        # store the previous action/reward to be shared across processes
        self._prev_action_ = np.frombuffer(Array('i', 2).get_obj(), dtype='i')

        # create factory with common arguments for making an observation dimension
        observation_factory = Create2ObservationFactory(main_op=self._main_op,
                                                        dt=dt,
                                                        obs_history=self._obs_history,
                                                        internal_timing=self._internal_timing,
                                                        prev_action=self._prev_action_)

        # the definition of the observed state and the associated custom modification (if any)
        # before passing to the learning algorithm
        self._observation_def = [
            observation_factory.make_dim('light bump left signal'),
            observation_factory.make_dim('light bump front left signal'),
            observation_factory.make_dim('light bump center left signal'),
            observation_factory.make_dim('light bump center right signal'),
            observation_factory.make_dim('light bump front right signal'),
            observation_factory.make_dim('light bump right signal'),
            observation_factory.make_dim('previous action')
        ]

        # extra packets we need for proper reset and charging
        self._extra_sensor_packets = ['bumps and wheel drops', 'battery charge',
                                      'oi mode', 'distance','charging sources available',
                                      'cliff left', 'cliff front left', 'cliff front right', 'cliff right']
        main_sensor_packet_ids = [d.packet_id for d in self._observation_def if d.packet_id is not None]
        extra_sensor_packet_ids = [create2_config.PACKET_NAME_TO_ID[nm] for nm in self._extra_sensor_packets]

        # TODO: move this out to some base class?
        from gym.spaces import Box as GymBox  # use this for baselines algos
        Box = GymBox

        # go thru the main opcode (just direct_drive in this case) and add the range of each param
        # XXX should the action space include the opcode? what about op that doesn't have parameters?
        self._action_space = Box(
            low=np.array([r[0] for r in create2_config.OPCODE_INFO[main_opcode]['params'].values()]),
            high=np.array([r[1] for r in create2_config.OPCODE_INFO[main_opcode]['params'].values()])
        )

        # loop thru the observation dimension and get the lows and highs
        self._observation_space = Box(
            low=np.concatenate([d.lows for d in self._observation_def]),
            high=np.concatenate([d.highs for d in self._observation_def])
        )

        # self._comm_name = 'Create2'
        communicator_setups = {}
        buffer_len = int(dt / self._internal_timing + 1)
        communicator_setups['Create2'] = {'Communicator': Create2Communicator,
                                                 # have to read in this number of packets everytime to support
                                                 # all operations
                                                 'num_sensor_packets': buffer_len,
                                                 'kwargs': {'sensor_packet_ids': main_sensor_packet_ids +
                                                                                 extra_sensor_packet_ids,
                                                            'opcodes': [main_opcode] + extra_opcodes,
                                                            'port': port,
                                                            'buffer_len': 2 * buffer_len,
                                                           }
                                            }
        

        self._roomba_obs_buffer = SharedBuffer(
                buffer_len=SharedBuffer.DEFAULT_BUFFER_LEN,
                array_len=len(self._observation_space.low)+2,
                array_type='d',
                np_array_type='d',
                )

        if image_shape != (0, 0, 0):
            image_stack = int(image_shape[0] // 3)
            communicator_setups['Camera'] = {'Communicator': CameraCommunicator,
                                             'num_sensor_packets': image_stack,
                                             'kwargs': 
                                                    {'device_id': camera_id,
                                                     'res': (image_shape[2], image_shape[1]) # communicator uses w, h
                                                    }
                                            }

            self._image_obs_buffer = SharedBuffer(
                buffer_len=SharedBuffer.DEFAULT_BUFFER_LEN,
                array_len=image_shape[0]*image_shape[1]*image_shape[2]+1,
                array_type='B',
                np_array_type='B',
                )
            self._image_reward_buffer = SharedBuffer(
                buffer_len=SharedBuffer.DEFAULT_BUFFER_LEN,
                array_len=1,
                array_type='d',
                np_array_type='d',
                )

        self._image_shape = image_shape
        self._image_space = Box(low=0, high=255, shape=self._image_shape)
        super().__init__(communicator_setups=communicator_setups,
                        action_dim=len(self._action_space.low),
                        observation_dim=-2, # dont use the base class sensation buffer
                        dt=dt,
                        **kwargs)
    
    def _sensor_to_sensation_(self):
        # overwrite this to support image
        for name, comm in self._sensor_comms.items():
            if comm.sensor_buffer.updated():
                sensor_window, timestamp_window, index_window = comm.sensor_buffer.read_update(self._num_sensor_packets[name])
                if name == 'Create2':
                    s = self._compute_roomba_obs_(sensor_window, timestamp_window, index_window)
                    self._roomba_obs_buffer.write(s, timestamp=timestamp_window[-1])
                elif name == 'Camera':
                    s, r = self._compute_image_obs_(sensor_window, timestamp_window, index_window)
                    self._image_obs_buffer.write(s, timestamp=timestamp_window[-1])
                    self._image_reward_buffer.write(r, timestamp=timestamp_window[-1])
                else:
                    raise NotImplementedError('Unsupported communicator')

    def _read_sensation(self):
        # overwrite this to support image
        roomba_obs_r_d, roomba_obs_timestamp, _ = self._roomba_obs_buffer.read_update()
        roomba_obs, r_r, r_d = roomba_obs_r_d[0][:-2], roomba_obs_r_d[0][-2], roomba_obs_r_d[0][-1]
        image = None
        im_r = 0
        im_d = 0
        if self._image_shape != (0, 0, 0):
            image_d, image_timestamp, _ = self._image_obs_buffer.read_update()
            image, im_d = image_d[0][:-1], image_d[0][-1]
            im_r, _, _ = self._image_reward_buffer.read_update()
            im_r = im_r[0][0]

            delay = abs(image_timestamp[-1] - roomba_obs_timestamp[-1])
            if delay > self._dt:
                print('Warning: image time and proprioception time is different by: {}s.'.format(delay))
                #print
            # unflatten image
            stacks = int(self._image_shape[0]//3)
            height = self._image_shape[1]
            width = self._image_shape[2]
            image = np.transpose(image.reshape((stacks, height, width, 3)), (0, 3, 1, 2)) # s, c, h, w
            image = np.concatenate(image, axis=0) # change to self._image_shape

        #print('r_r:', r_r, "im_r:", im_r)
        done = self._check_done(r_d or im_d)
        return (image, roomba_obs), r_r+im_r-1,  done

    def _compute_image_obs_(self, sensor_window, timestamp_window, index_window):
        # return np.concatenate((actual_sensation, [reward], [done]))
        reward, done = self._calc_image_reward(sensor_window)
        flatten_image = np.concatenate(sensor_window, axis=0)

        return np.concatenate((flatten_image, [done])).astype('uint8'), reward

    def _compute_roomba_obs_(self, sensor_window, timestamp_window, index_window):
        """The required _computer_sensation_ interface.
        Args:
            name:               the name of communicator the sense is from
            sensor_window:      an array of size num_sensor_packets each containing 1 complete observation packets
            timestamp_window:   array of timestamp corresponds to the sensor_window
            index_window:       array of count corresponds to the sensor_window
        Returns:
            A numpy array with [:-2] the sensation, [-2] the reward, [-1] the done flag
        """
        # construct the actual sensation

        actual_obs = []
        for d in self._observation_def:
            res = d.normalized_handler(sensor_window)
            actual_obs.extend(res)

        # accumulate the rotation information
        # self._total_rotation += sensor_window[-1][0]['angle']

        reward, done = self._calc_roomba_obs_reward(sensor_window)

        return np.concatenate((actual_obs, [reward], [done]))

    def _compute_actuation_(self, action, timestamp, index):
        """The required _compute_actuator_ interface.
        The side effect is to write the output to self._actuation_packet_[name] with [opcode, *action]
        Args:
            action:      an array of 2 numbers correspond to the speed of the left & right wheel
            timestamp:   the timestamp when the action was written to buffer
            index:       the action count
        """
        # add a safety check for any action with nan or inf
        if any([not np.isfinite(a) for a in action]):
            logging.warning("Invalid action received: {}".format(action))
            return

        # pass int only action
        action = action.astype('i')
        self._actuation_packet_['Create2'] = np.concatenate(
            ([create2_config.OPCODE_NAME_TO_CODE[self._main_op]], action))
        np.copyto(self._prev_action_, action)

    def _reset_(self):
        """The required _reset_ interface.
        This method does the handling of charging the Create2, repositioning, and set to the correct mode.
        """
        logging.info("Resetting...")
        self._episode_step_.value = -1
        np.copyto(self._prev_action_, np.array([0, 0]))
        for d in self._observation_def:
            d.reset()

        # wait for create2 to startup properly if just started (ie. wait to actually start receiving observation)
        while not self._sensor_comms['Create2'].sensor_buffer.updated():
            time.sleep(0.01)

        # wait for camera to startup properly
        if self._image_shape != (0, 0, 0):
            while not self._sensor_comms['Camera'].sensor_buffer.updated():
                time.sleep(0.01)

        sensor_window, _, _ = self._sensor_comms['Create2'].sensor_buffer.read()
        print('Current charge:', sensor_window[-1][0]['battery charge'])
        if sensor_window[-1][0]['battery charge'] <= self._min_battery:
            print('Recharging...')
            self._seek_charger()
            sensor_window, _, _ = self._sensor_comms['Create2'].sensor_buffer.read()
            while sensor_window[-1][0]['battery charge'] < self._max_battery:
                print('Current charge:', sensor_window[-1][0]['battery charge'])
                logging.info("Create2 charging with current charge at {}.".format(sensor_window[-1]['battery charge']))
                time.sleep(10)
                sensor_window, _, _ = self._sensor_comms['Create2'].sensor_buffer.read()

        # Always switch to SAFE mode to run an episode, so that Create2 will switch to PASSIVE on the
        # charger.  If the create2 is in any other mode on the charger, we will not be able to detect
        # the non-responsive sleep mode that happens at the 60 seconds mark.
        logging.info("Setting Create2 into safe mode.")
        self._write_opcode('safe')
        time.sleep(0.1)

        # go back to the charger if not on it
        sensor_window, _, _ = self._sensor_comms['Create2'].sensor_buffer.read()
        if sensor_window[-1][0]['charging sources available'] <= 0:
            # go to a random location
            print("Moving Create2 into a random position.")
            target_values = [-150, -150]
            move_time = np.random.uniform(low=1, high=1.5)
            
            # back
            self._write_opcode('drive_direct', *target_values)
            time.sleep(move_time)
            self._write_opcode('drive', 0, 0)
            time.sleep(0.1)

            self._seek_charger()

        # after charging/docked, try to drive away from the dock if still on it
        sensor_window, _, _ = self._sensor_comms['Create2'].sensor_buffer.read()
        if sensor_window[-1][0]['charging sources available'] > 0:
            print("Undocking the Create2.")
            self._write_opcode('drive_direct', -150, -150)
            t = np.random.uniform(0, 1)
            time.sleep(t)
            self._write_opcode('drive_direct', 0, 0)
            time.sleep(0.1)

        # make sure in SAFE mode in case the random drive caused switch to PASSIVE, or
        # create2 stuck somewhere and require human reset (don't want an episode to start
        # until fixed, otherwise we get a whole bunch of one step episodes)
        sensor_window, _, _ = self._sensor_comms['Create2'].sensor_buffer.read()
        while sensor_window[-1][0]['oi mode'] != 2:
            logging.warning("Create2 not in SAFE mode, reattempting... (might require human intervention).")
            self._write_opcode('full')
            time.sleep(0.2)
            self._write_opcode('drive_direct', -50, -50)
            time.sleep(0.5)
            self._write_opcode('drive', 0, 0)
            time.sleep(0.1)
            self._write_opcode('safe')
            time.sleep(0.2)
            sensor_window, _, _ = self._sensor_comms['Create2'].sensor_buffer.read()

        # don't want to state during reset pollute the first sensation
        time.sleep(1)
        # sensor_window, _, _ = self._sensor_comms['Create2'].sensor_buffer.read()
        # while sensor_window[-1][0]['charging sources available'] > 0:
        #     time.sleep(1)
        #     sensor_window, _, _ = self._sensor_comms['Create2'].sensor_buffer.read()

        print("Reset completed.")

    def _seek_charger(self):
        self._write_opcode('safe')
        time.sleep(0.1)
        print('Seeking charger')
        self._write_opcode('seek_dock')
        sensor_window, _, _ = self._sensor_comms['Create2'].sensor_buffer.read()
        while sensor_window[-1][0]['charging sources available'] <= 0:
            time.sleep(1)
            sensor_window, _, _ = self._sensor_comms['Create2'].sensor_buffer.read()

        self._write_opcode('safe')
        time.sleep(0.1)

    def _check_done(self, env_done):
        """The required _check_done_ interface.
        Args:
            env_done:   whether the environment is done from _compute_sensation_
        Returns:
            A boolean flag for done
        """
        self._episode_step_.value += 1
        return self._episode_step_.value >= self._episode_length_step or env_done

    def _calc_roomba_obs_reward(self, sensor_window):
        """Helper to calculate reward.
        Args:
            sensor_window: the sensor_window from _compute_sensation_
        Returns:
            A tuple of (reward, done)
        """
        
        bw = 0
        for p in range(int(self._dt / self._internal_timing)):
            bw |= sensor_window[-1 - p][0]['bumps and wheel drops']
        
        cl = 0
        for p in range(int(self._dt / self._internal_timing)):
            cl += sensor_window[-1 - p][0]['cliff left']
            cl += sensor_window[-1 - p][0]['cliff front left']
            cl += sensor_window[-1 - p][0]['cliff front right']
            cl += sensor_window[-1 - p][0]['cliff right']

        reward = 0.0
        
        charging_sources_available = sensor_window[-1][0]['charging sources available']

        oi_mode = sensor_window[-1][0]['oi mode']
        if oi_mode == 1 and charging_sources_available == 0 and cl == 0:
            self._write_opcode('safe')

        done = int(charging_sources_available > 0)

        # If wheel dropped, it's done and result in a big penalty.
        if (bw >> 2) > 0:
            done = 1
            reward = -self._episode_length_step

        return reward, done

    def _calc_image_reward(self, sensor_window):
        reward = 0.0
        done = 0

        return reward, done

    def _write_opcode(self, opcode_name, *args):
        """Helper method to force write a command not part of the action dimension.
        Args:
            opcode_name:    the name of the opcode
            *args:          any arguments require for the operation
        """
        # write the command directly to actuator_buffer to avoid the limitation that the opcode
        # is not part of the action dimension
        self._actuator_comms['Create2'].actuator_buffer.write(
            np.concatenate(([create2_config.OPCODE_NAME_TO_CODE[opcode_name]], np.array(args).astype('i'))))

    # ======== rllab compatible gym codes =========

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def image_space(self):
        return self._image_space

    @property
    def proprioception_space(self):
        return self._observation_space

    def terminate(self):
        super().close()

if __name__ == '__main__':
    episode_length_step = int(30 / 0.045)

    env = Create2VisualDockerEnv(episode_length_time=30, dt=0.045, image_shape=(9, 120, 160), camera_id=0)
    env.start()
    env.reset()

    episode_step = 0
    episode_ret = 0
    episode = 0
    for i in range(100000):
        a = env.action_space.sample()
        (image, obs), reward, done, _ = env.step(a)

        episode_step += 1
        episode_ret += reward
        '''
        image = np.transpose(image, [1, 2, 0])
        cv2.imshow('', image[:,:,0:3])
        cv2.waitKey(1)
        '''

        if done:
            episode += 1
            print(f'episode: {episode}, return: {episode_ret}, episode_step: {episode_step}, step: {i}')
    
            env.reset()
            episode_step = 0
            episode_ret = 0
