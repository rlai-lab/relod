#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Jun Jin
# Created Date: Thu May 14 MDT 2020
# Revised Date: N/A
# =============================================================================
"""
utils functions
Copyright (c) 2019, Huawei Canada Inc.
All rights reserved.
"""

import os
from gym.logger import error
import numpy as np
import collections
import math
import random
from collections import namedtuple
import torch
from PIL import Image
import pickle
import torch.utils.data as Data
from torchvision import datasets, models, transforms
import time
import yaml
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import time
import tf


def configure(config_file):
    """
    Opens and reads the config file and stores the data in dictionary form as an instance attribute.

    :param config_file: (.yaml) file consisting of simulation config
    """
    print(config_file)
    with open(config_file) as config:
        try:
            return yaml.safe_load(config)
        except Exception as err:
            print('Error Configuration File:{}'.format(err))
            raise err


def grey_scale_img_norm(grey_scale_img):
    obs = grey_scale_img/255
    obs += -(np.min(obs))
    obs_max = np.max(obs)
    if obs_max != 0:
        obs /= np.max(obs) / 2
    obs += -1
    return np.expand_dims(obs,0)


def smoothly_move_to_position(robot, target_joints, control_frequency=30, motion_duration=4):
    initial_pose = robot.joint_angles()  # get current joint angles of the robot
    # print(initial_pose)
    max_iterations = int(motion_duration * control_frequency)
    for i in range(max_iterations):
        vals = robot.joint_angles()
        elapsed_time = i/control_frequency
        ratio = math.exp(2*elapsed_time - 4)/(math.exp(2*elapsed_time - 4)+1)   # ration [0, 1]
        for j in robot.joint_names():
            vals[j] = initial_pose[j] + (target_joints[j] - initial_pose[j]) * ratio
        robot.set_joint_positions(vals) # set joint positions for the robot.
        time.sleep(1/control_frequency)
    # fine tuning for accuracy, TODO
    return True


def smoothly_move_to_position_vel(robot, robot_status, target_joints, control_frequency=40, motion_duration=4,MAX_JOINT_VELs = .6):
    initial_pose = robot.joint_angles()  # get current joint angles of the robot
    #print(initial_pose)
    max_iterations = int(motion_duration * control_frequency)
    robot_status.enable()
    
    t0 = time.time()
    stop_counter = 0
    ratio = 2
    for _ in range(max_iterations):
        vals = robot.joint_angles()
        errors = robot.joint_angles() 
        #ratio = 2
        #print("current joints", vals)
        for j in robot.joint_names():
            errors[j] = (target_joints[j] - vals[j])
            if np.abs(errors[j]) < 0.05:
                ratio = 2#np.abs(40*errors[j])
                #print('ration is ', ratio)
            else:
                ratio = 2
            vals[j] = (target_joints[j] - vals[j])*ratio
            vals[j] = np.clip(vals[j], -MAX_JOINT_VELs, MAX_JOINT_VELs)
        # print(vals)
        robot.set_joint_velocities(vals) # set joint positions for the robot.
        #robot.set_joint_positions(target_joints[j])
        time.sleep(1/control_frequency)
        ## added this line to finish earlier
        if max([np.abs(vals[j]) for j in robot.joint_names()]) < 0.01:
            ratio = 0
            for j in robot.joint_names():
            
                vals[j] = 0
                # print(vals)
                robot.set_joint_velocities(vals)
            break
        robot_status.enable()
    time.sleep(0.2)
    robot_status.enable()
    # print(time.time()-t0 )
    return True


def scalar_out_of_range(scalar, range):
    if scalar < range[0] or scalar > range[1]:
        return True
    else:
        return False


def random_val_continous(arg):
    """
    Will check if a value should be random, and randoimize it if so.

    Parameters
    ----------
    arg : list or float
        The value to possibly be randomized
    Returns
    -------
    float
        If arg is a list it is a random number between arg[0] and arg[1],
        otherwise it is arg.
    """
    if isinstance(arg,list):
        return random.uniform(*arg)
    else:
        return arg


def pbvs6(current_pos, ref_pos, _lambda_xyz=1.2, _lambda_orn=1.2):
    current_xyz = np.array(current_pos[0:3])
    ref_xyz = np.array(ref_pos[0:3])  # + [0,0,0.005]
    residual_error = np.linalg.norm(ref_xyz - current_xyz)
    if residual_error<0.008:
        ref_xyz = np.array(ref_pos[0:3]) - [0, 0, 0.001]
    error_xyz = (ref_xyz - current_xyz)*_lambda_xyz
    current_qxyzw = current_pos[3:7]
    ref_qxyzw = ref_pos[3:7]
    # Compute the rotation in angle-axis format that rotates next_qxyzw into current_qxyzw.
    (x, y, z, angle) = pr.quaternion_diff(current_qxyzw, ref_qxyzw)
    error_orn = (-1) * _lambda_orn * angle * np.array([x, y, z])
    return np.hstack((error_xyz, error_orn))


def extract_values(dict, ordered_keys):
    values = []
    for key in ordered_keys:
        values.append(dict[key])
    return values


def generate_random_target_robot_pose(bullet_client, slot_id, yaw_bound=[-math.pi, math.pi]):
    slot_pos, _ = bullet_client.getBasePositionAndOrientation(slot_id)
    x, y, z = slot_pos
    z = 0.9
    orn_euler = [math.pi, 0, random.uniform(yaw_bound[0], yaw_bound[1])]
    upright_orientation = bullet_client.getQuaternionFromEuler(orn_euler)
    bullet_client.addUserDebugLine([x, y, z - 0.5], [x, y, z], lineColorRGB=[1, 0, 0])
    return [x, y, z], upright_orientation


def generate_skew_mat(v):
    """
    Returns the corresponding skew symmetric matrix from a 3-vector
    """
    skew_matrix=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return skew_matrix


def change_qxyzw2qwxyz(q):
    new_q = np.concatenate((q[3:4], q[0:3]), axis=0)
    return new_q

