# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import time
import gym
import sys
from multiprocessing import Array, Value
import json

from senseact.envs.ur.reacher_env import ReacherEnv
from senseact.devices.ur import ur_utils
from senseact.devices.ur.ur_setups import setups
from senseact.sharedbuffer import SharedBuffer
from senseact import utils


class ReacherEnvMinTime(ReacherEnv):
    """A class implementing UR5 Reacher2D and Reacher6D environments.

    A UR5 reacher task consists in reaching with an arm end-effector a given
    location in Cartesian space using low level UR5 joint control commands.
    The class implements a fixed duration episodic reacher task where the target is
    generated randomly at the beginning of each episode. The class implements
    both, position and velocity control. For each of those it implements direct,
    first and second derivative control. The class implements safety boundaries
    handling defined as a box in a Cartesian space. The class also implements reset
    function, which moves the arm into a predefined initial position at the end
    of each episode and generates a random target.
    """
    def __init__(self,
                 setup,
                 host=None,
                 dof=6,
                 control_type='position',
                 derivative_type='none',
                 target_type='position',
                 reset_type='random',
                 deriv_action_max=10,
                 first_deriv_max=10,  # used only with second derivative control
                 vel_penalty=0,
                 obs_history=1,
                 actuation_sync_period=1,
                 episode_length_time=None,
                 episode_length_step=None,
                 rllab_box = False,
                 servoj_t=ur_utils.COMMANDS['SERVOJ']['default']['t'],
                 servoj_gain=ur_utils.COMMANDS['SERVOJ']['default']['gain'],
                 speedj_a=ur_utils.COMMANDS['SPEEDJ']['default']['a'],
                 speedj_t_min=ur_utils.COMMANDS['SPEEDJ']['default']['t_min'],
                 movej_t=2, # used for resetting
                 accel_max=None,
                 speed_max=None,
                 dt=0.008,
                 delay=0.0,  # to simulate extra delay in the system
                 **kwargs):
        """Inits ReacherEnv class with task and robot specific parameters.

        Args:
            setup: a dictionary containing UR5 reacher task specifications,
                such as safety box dimensions, joint angle ranges, boundary
                on the arm speed, UR5 Controller IP address etc
                (see senseact.devices.ur.ur_setups for examples).
            host: a string specifying UR5 IP address or None
            dof: an integer number of degrees of freedom, either 2 for
                Reacher2D or 6 for Reacher6D
            control_type: a string specifying UR5 control type, either
                position (using UR5 servoJ commands) or velocity
                (using UR5 speedJ commands)
            derivative_type: a string specifying what type of derivative
                control to use, either "none", "first" or "seconds"
            target_type: a string specifying in what space to provide
                target coordinates, either "position" for Cartesian space
                or "angle" for joints angles space.
            reset_type: a string specifying whether to reset the arm to a
                fixed position or to a random position.
            reward_type: a string specifying the reward function, either
                "linear" for - d_t, or "precision" for  - d_t + exp^( - d_t^2)
            deriv_action_max: a float specifying maximum value of an action
                for derivative control
            first_deriv_max: a float specifying maximum value of a first
                derivative of action if derivative_type =="second"
            vel_penalty: a float specifying the weight of a velocity
                penalty term in the reward function.
            obs_history: an integer number of sensory packets concatenated
                into a single observation vector
            actuation_sync_period: a bool specifying whether to synchronize
                sending actuation commands to UR5 with receiving sensory
                packets from UR5 (must be true for smooth UR5 operation).
            episode_length_time: a float duration of an episode defined
                in seconds
            episode_length_step: an integer duration of en episode
                defined in environment steps.
            rllab_box: a bool specifying whether to wrap environment
                action and observation spaces into an RllabBox object
                (required for off-the-shelf rllab algorithms implementations).
            servoj_t: a float specifying time parameter of a UR5
                servoj command.
            servoj_gain: a float specifying gain parameter of a UR5
                servoj command.
            speedj_a: a float specifying acceleration parameter of a UR5
                speedj command.
            speedj_t_min: a float specifying t_min parameter of a UR5
                speedj command.
            movej_t: a float specifying time parameter of a UR5
                speedj command.
            accel_max: a float specifying maximum allowed acceleration
                of UR5 arm joints. If None, a value from setup is used.
            speed_max: a float specifying maximum allowed speed of UR5 joints.
                If None, a value from setup is used.
            dt: a float specifying duration of an environment time step
                in seconds.
            delay: a float specifying artificial observation delay in seconds

        """


        # Check that the task parameters chosen are implemented in this class
        assert dof in [2] # not tested for 6 dof yet
        assert control_type in ['position', 'velocity', 'acceleration']
        assert derivative_type in ['none', 'first', 'second']
        assert target_type in ['position', 'angle']
        assert reset_type in ['random', 'zero', 'none']
        assert actuation_sync_period >= 0

        if episode_length_step is not None:
            assert episode_length_time is None
            self._episode_length_step = episode_length_step
            self._episode_length_time = episode_length_step * dt
        elif episode_length_time is not None:
            assert episode_length_step is None
            self._episode_length_time = episode_length_time
            self._episode_length_step = int(episode_length_time / dt)
        else:
            #TODO: should we allow a continuous behaviour case here, with no episodes?
            print("episode_length_time or episode_length_step needs to be set")
            raise AssertionError

        # Task Parameters
        self._host = setups[setup]['host'] if host is None else host
        self._obs_history = obs_history
        self._dof = dof
        self._control_type = control_type
        self._derivative_type = derivative_type
        self._target_type = target_type
        self._reset_type = reset_type
        self._vel_penalty = vel_penalty # weight of the velocity penalty
        self._deriv_action_max = deriv_action_max
        self._first_deriv_max = first_deriv_max
        self._speedj_a = speedj_a
        self._delay = delay
        self._comm_history = 10
        self._safety_violation_count = 0
        self.return_point = None
        if accel_max==None:
            accel_max = setups[setup]['accel_max']
        if speed_max==None:
            speed_max = setups[setup]['speed_max']
        if self._dof == 6:
            self._joint_indices = [0, 1, 2, 3, 4, 5]
            self._end_effector_indices = [0, 1, 2]
        elif self._dof == 2:
            self._joint_indices = [1, 2]
            self._end_effector_indices = [1, 2]

        # Arm/Control/Safety Parameters
        self._pos_target_low = setups[setup]['pos_target_low']
        self._pos_target_high = setups[setup]['pos_target_high']
        self._end_effector_low = setups[setup]['end_effector_low']
        self._end_effector_high = setups[setup]['end_effector_high']
        self._angles_low = setups[setup]['angles_low'][self._joint_indices]
        self._angles_high = setups[setup]['angles_high'][self._joint_indices]
        self._speed_low = -np.ones(self._dof) * speed_max
        self._speed_high = np.ones(self._dof) * speed_max
        self._accel_low = -np.ones(self._dof) * accel_max
        self._accel_high = np.ones(self._dof) * accel_max

        self._box_bound_buffer = setups[setup]['box_bound_buffer']
        self._angle_bound_buffer = setups[setup]['angle_bound_buffer']
        self._q_ref = setups[setup]['q_ref']
        self._ik_params = setups[setup]['ik_params']

        # State Variables
        self._q_ = np.zeros((self._obs_history, self._dof))
        self._qd_ = np.zeros((self._obs_history, self._dof))

        self._episode_steps = 0
        self._in_safety_violation = False

        self._pstop_time_ = None
        self._pstop_times_ = []
        self._safety_mode_ = ur_utils.SafetyModes.NONE
        self._max_pstop = 10
        self._max_pstop_window = 600
        self._clear_pstop_after = 2
        self._x_target_ = np.frombuffer(Array('f', 3).get_obj(), dtype='float32')
        self._x_ = np.frombuffer(Array('f', 3).get_obj(), dtype='float32')
        self._reward_ = Value('d', 0.0)

        if self._target_type == 'position':
            if self._dof == 2:
                self._target_ = np.zeros((2))
                self._target_low = self._pos_target_low[self._end_effector_indices]
                self._target_high = self._pos_target_high[self._end_effector_indices]
            elif self._dof == 6:
                self._target_ = np.zeros((3))
                self._target_low = self._pos_target_low
                self._target_high = self._pos_target_high
        elif self._target_type == 'angle':
            self._target_ = np.zeros((self._dof))
            self._target_low = self._angles_low
            self._target_high = self._angles_high

        # Set up action and observation space

        if self._derivative_type== 'second' or self._derivative_type== 'first':
            self._action_low = -np.ones(self._dof) * self._deriv_action_max
            self._action_high = np.ones(self._dof) * self._deriv_action_max
        else: # derivative_type=='none'
            if self._control_type == 'position':
                self._action_low = self._angles_low
                self._action_high = self._angles_high
            elif self._control_type == 'velocity':
                self._action_low = self._speed_low
                self._action_high = self._speed_high
            elif self._control_type == 'acceleration':
                self._action_low = self._accel_low
                self._action_high = self._accel_high

        # TODO: is there a nicer way to do this?
        if rllab_box:
            from rllab.spaces import Box as RlBox  # use this for rllab TRPO
            Box = RlBox
        else:
            from gym.spaces import Box as GymBox  # use this for baselines algos
            Box = GymBox

        self._observation_space = Box(
            low=np.array(
                list(self._angles_low * self._obs_history)  # q_actual
                + list(-np.ones(self._dof * self._obs_history))  # qd_actual
                + list(self._target_low)  # target
                + list(-self._action_low)  # previous action in cont space
            ),
            high=np.array(
                list(self._angles_high * self._obs_history)  # q_actual
                + list(np.ones(self._dof * self._obs_history))  # qd_actual
                + list(self._target_high)  # target
                + list(self._action_high)    # previous action in cont space
            )
        )


        self._action_space = Box(low=self._action_low, high=self._action_high)

        if rllab_box:
            from rllab.envs.env_spec import EnvSpec
            self._spec = EnvSpec(self.observation_space, self.action_space)

        # Only used with second derivative control
        self._first_deriv_ = np.zeros(len(self.action_space.low))

        # Communicator Parameters
        communicator_setups = {'UR5':
                                   {
                                    'num_sensor_packets': self._comm_history,

                                    'kwargs': {'host': self._host,
                                               'actuation_sync_period': actuation_sync_period,
                                               'speedj_timeout': 0.2,
                                               'buffer_len': self._comm_history + SharedBuffer.DEFAULT_BUFFER_LEN,
                                               }
                                    }
                               }
        if self._delay > 0:
            from senseact.devices.ur.ur_communicator_delay import URCommunicator
            communicator_setups['UR5']['kwargs']['delay'] = self._delay
        else:
            from senseact.devices.ur.ur_communicator import URCommunicator
        communicator_setups['UR5']['Communicator'] = URCommunicator

        super(ReacherEnv, self).__init__(communicator_setups=communicator_setups,
                                         action_dim=len(self.action_space.low),
                                         observation_dim=len(self.observation_space.low),
                                         dt=dt,
                                         **kwargs)

        # The last action
        self._action_ = self._rand_obj_.uniform(self._action_low, self._action_high)

        # Defining packet structure for quickly building actions
        self._reset_packet = np.ones(self._actuator_comms['UR5'].actuator_buffer.array_len) * ur_utils.USE_DEFAULT
        self._reset_packet[0] = ur_utils.COMMANDS['MOVEJ']['id']
        self._reset_packet[1:1 + 6] = self._q_ref
        self._reset_packet[-2] = movej_t

        self._servoj_packet = np.ones(self._actuator_comms['UR5'].actuator_buffer.array_len) * ur_utils.USE_DEFAULT
        self._servoj_packet[0] = ur_utils.COMMANDS['SERVOJ']['id']
        self._servoj_packet[1:1 + 6] = self._q_ref
        self._servoj_packet[-3] = servoj_t
        self._servoj_packet[-1] = servoj_gain

        self._speedj_packet = np.ones(self._actuator_comms['UR5'].actuator_buffer.array_len) * ur_utils.USE_DEFAULT
        self._speedj_packet[0] = ur_utils.COMMANDS['SPEEDJ']['id']
        self._speedj_packet[1:1 + 6] = np.zeros((6,))
        self._speedj_packet[-2] = speedj_a
        self._speedj_packet[-1] = speedj_t_min

        self._stopj_packet = np.zeros(self._actuator_comms['UR5'].actuator_buffer.array_len)
        self._stopj_packet[0] = ur_utils.COMMANDS['STOPJ']['id']
        self._stopj_packet[1] = 2.0

        # Tell the arm to do nothing (overwritting previous command)
        self._nothing_packet = np.zeros(self._actuator_comms['UR5'].actuator_buffer.array_len)

        self._pstop_unlock_packet = np.zeros(self._actuator_comms['UR5'].actuator_buffer.array_len)
        self._pstop_unlock_packet[0] = ur_utils.COMMANDS['UNLOCK_PSTOP']['id']


        # self.info['reward_dist'] = 0
        # self.info['reward_vel'] = 0
        # debug fix target
        reset_angles, _ = self._pick_random_angles_()
        self._q_target_, x_target = self._pick_random_target_(reset_angles)
        np.copyto(self._x_target_, x_target)
        if self._target_type == 'position':
            self._target_ = self._x_target_[self._end_effector_indices]
        elif self._target_type == 'angle':
            self._target_ = self._q_target_

    def _reset_(self):
        """Resets the environment episode.

        Moves the arm to either fixed reference or random position and
        generates a new target within a safety box.
        """
        print("Resetting")

        self._action_ = self._rand_obj_.uniform(self._action_low, self._action_high)
        self._cmd_prev_ = np.zeros(len(self._action_low))  # to be used with derivative control of velocity
        if self._reset_type != 'none':
            if self._reset_type == 'random':
                reset_angles, _ = self._pick_random_angles_()
            elif self._reset_type == 'zero':
                reset_angles = self._q_ref[self._joint_indices]
            self._reset_arm(reset_angles)

        self._q_target_, x_target = self._pick_random_target_(reset_angles)
        np.copyto(self._x_target_, x_target)
        if self._target_type == 'position':
            self._target_ = self._x_target_[self._end_effector_indices]
        elif self._target_type == 'angle':
            self._target_ = self._q_target_

        rand_state_array_type, rand_state_array_size, rand_state_array = utils.get_random_state_array(
            self._rand_obj_.get_state()
        )
        np.copyto(self._shared_rstate_array_, np.frombuffer(rand_state_array, dtype=rand_state_array_type))

        print("Reset done")

    def _pick_random_angles_(self):
        """Generates a set of random angle positions for each joint."""
        movej_q = self._q_ref.copy()
        while True:
            reset_angles = self._rand_obj_.uniform(self._angles_low, self._angles_high)
            movej_q[self._joint_indices] = reset_angles
            inside_bound, inside_buffer_bound, mat, xyz = self._check_bound(movej_q)
            inside_target_bound = np.all(self._pos_target_low <= xyz) and np.all(xyz <= self._pos_target_high)
            if inside_buffer_bound and inside_target_bound:
                break
        return reset_angles, xyz

    def _pick_random_target_(self, reset_angles):
        """Generates a set of random target coordinates."""
        movej_q = self._q_ref.copy()
        while True:
            target_angles = self._rand_obj_.uniform(self._angles_low, self._angles_high)
            movej_q[self._joint_indices] = target_angles
            inside_bound, inside_buffer_bound, mat, xyz = self._check_bound(movej_q)
            inside_target_bound = np.all(self._pos_target_low <= xyz) and np.all(xyz <= self._pos_target_high)
            if not inside_buffer_bound or not inside_target_bound:
                continue

#             mat = ur_utils.forward(movej_q, self._ik_params)
#             x_target = mat[:3, 3].copy()
#             #x_target = np.zeros_like(self._x_target_)
#             x_target[self._end_effector_indices] = self._rand_obj_.uniform(self._target_low, self._target_high)
#             proj_x_target = x_target.copy()
#             x_target_norm = np.linalg.norm(proj_x_target)
#             #print(proj_x_target)
#             #print(x_target_norm)
#             radius = 0.95
#             if x_target_norm >= radius:
#             denom = np.sqrt(np.sum(np.power(proj_x_target, 2)) / radius**2)
#             #print(denom.shape)
#             if self._dof == 2:
#                 denom = np.sqrt(np.sum(np.power(proj_x_target[self._end_effector_indices], 2)) / (radius**2 - np.power(proj_x_target[0], 2)))
#             proj_x_target[self._end_effector_indices] /= denom

            #print(proj_x_target)
            #print(np.linalg.norm(proj_x_target))
            mat[self._end_effector_indices, 3] = xyz[self._end_effector_indices]
            #print(mat[:3, 3])
            ref_pos = self._q_ref.copy()
            ref_pos[self._joint_indices] = reset_angles
            solutions = ur_utils.inverse_near(mat, wrist_desired=self._q_ref[-1], ref_pos=ref_pos,
                      params=self._ik_params)
            if not solutions:
                continue

            target_found = False
            for solution in solutions:
                q_target = self._q_ref.copy()
                q_target[self._joint_indices] = solution[self._joint_indices]
                #print(q_target)
                inside_bound, inside_buffer_bound, mat, x_target = self._check_bound(q_target)
                inside_target_bound = np.all(self._pos_target_low <= x_target) and np.all(x_target <= self._pos_target_high)
                inside_angle_bound = np.all(self._angles_low <= q_target[self._joint_indices]) and \
                                 np.all(q_target[self._joint_indices] <= self._angles_high)

                if inside_buffer_bound and inside_target_bound and inside_angle_bound:
                    return q_target[self._joint_indices], x_target

        return target_angles, xyz


    def _compute_sensation_(self, name, sensor_window, timestamp_window, index_window):
        """Creates and saves an observation vector based on sensory data.

        For reacher environments the observation vector is a concatenation of:
            - current joint angle positions;
            - current joint angle velocities;
            - diference between current end-effector and target
              Cartesian coordinates;
            - previous action;

        Args:
            name: a string specifying the name of a communicator that
                received given sensory data.
            sensor_window: a list of latest sensory observations stored in
                communicator sensor buffer. the length of list is defined by
                obs_history parameter.
            timestamp_window: a list of latest timestamp values stored in
                communicator buffer.
            index_window: a list of latest sensor index values stored in
                communicator buffer.

        Returns:
            A numpy array containing concatenated [observation, reward, done]
            vector.
        """
        index_end = len(sensor_window)
        index_start = index_end - self._obs_history
        self._q_ = np.array([sensor_window[i]['q_actual'][0] for i in range(index_start,index_end)])
        self._qt_ = np.array([sensor_window[i]['q_target'][0] for i in range(index_start,index_end)])
        self._qd_ = np.array([sensor_window[i]['qd_actual'][0] for i in range(index_start,index_end)])
        self._qdt_ = np.array([sensor_window[i]['qd_target'][0] for i in range(index_start,index_end)])
        self._qddt_ = np.array([sensor_window[i]['qdd_target'][0] for i in range(index_start,index_end)])

        self._current_ = np.array([sensor_window[i]['i_actual'][0] for i in range(index_start,index_end)])
        self._currentt_ = np.array([sensor_window[i]['i_target'][0] for i in range(index_start,index_end)])
        self._currentc_ = np.array([sensor_window[i]['i_control'][0] for i in range(index_start,index_end)])
        self._mt_ = np.array([sensor_window[i]['m_target'][0] for i in range(index_start,index_end)])
        self._voltage_ = np.array([sensor_window[i]['v_actual'][0] for i in range(index_start,index_end)])

        self._safety_mode_ = np.array([sensor_window[i]['safety_mode'][0] for i in range(index_start,index_end)])
        if self._safety_mode_ != ur_utils.SafetyModes.NONE and \
           self._safety_mode_ != ur_utils.SafetyModes.NORMAL and \
           not self._in_safety_violation:
            self._in_safety_violation = True
            self._safety_violation_count += 1
        elif self._safety_mode_ == ur_utils.SafetyModes.NONE or \
             self._safety_mode_ == ur_utils.SafetyModes.NORMAL:
            self._in_safety_violation = False

        #TODO: should there be checks for safety modes greater than pstop here, and exit if found?

        # Compute end effector position
        x = ur_utils.forward(sensor_window[-1]['q_actual'][0], self._ik_params)[:3, 3]
        np.copyto(self._x_, x)

        if self._target_type == 'position':
            self._target_diff_ = self._x_[self._end_effector_indices] - self._target_
        elif self._target_type == 'angle':
            self._target_diff_ = self._q_[-1, self._joint_indices] - self._target_

        self._reward_.value = self._compute_reward_()
        dist = np.linalg.norm(self._target_diff_, ord=2)
        if dist <= 0.1:
            done = 1
        else:
            done = 0
        # TODO: use the correct obs that matches the observation_space
        return np.concatenate((self._q_[:, self._joint_indices].flatten(),
                               self._qd_[:, self._joint_indices].flatten() / self._speed_high,
                               self._target_,
                               self._action_ / self._action_high,
                               [self._reward_.value],
                               [done]))

    def _compute_reward_(self):
        """Computes reward at a given time step.

        Returns:
            A float reward.
        """

        # TODO: doublecheck whether '0' or '-1' should be used as the index
        reward_vel = -self._vel_penalty * np.square(self._qd_[-1, self._joint_indices]).sum()

        #self.info['reward_dist'] = reward_dist
        #self.info['reward_vel'] = reward_vel

        return -1
