# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Contains setups for UR reacher environments. Specifies safety
box dimensions, joint limits to avoid self-collision etc."""

import numpy as np

setups = {
    'UR5_default':
              {
                  'host': '192.168.2.152',  # put UR5 Controller address here
                  'end_effector_low': np.array([-0.2, -0.3, 0.5]),
                  'end_effector_high': np.array([0.2, 0.4, 1.0]),
                  'angles_low':np.pi/180 * np.array(
                      [ 60,
                       -180,#-180
                       -120,
                       -50,
                        50,
                        50
                       ]
                  ),
                  'angles_high':np.pi/180 * np.array(
                      [ 90,
                       -60,
                        130,
                        25,
                        120,
                        175
                       ]
                  ),
                  'speed_max': 0.3,   # maximum joint speed magnitude using speedj
                  'accel_max': 1,      # maximum acceleration magnitude of the leading axis using speedj
                  'reset_speed_limit': 0.5,
                  'q_ref': np.array([ 1.58724391, -2.4, 1.5, -0.71790582, 1.63685572, 1.00910473]),
                  'box_bound_buffer': 0.001,
                  'angle_bound_buffer': 0.001,
                  'ik_params':
                      (
                          0.089159, # d1
                          -0.42500, # a2
                          -0.39225, # a3
                          0.10915,  # d4
                          0.09465,  # d5
                          0.0823    # d6
                      )
              },
    'UR5_6dof':
              {
                  'host': '192.168.2.152',  # put UR5 Controller address here
                  'end_effector_low': np.array([-0.3, -0.6, 0.5]),
                  'end_effector_high': np.array([0.2, 0.4, 1.0]),
                  'angles_low':np.pi/180 * np.array(
                      [ 30,
                       -180,#-180
                       -120,
                       -120,
                        30,
                        0
                       ]
                  ),
                  'angles_high':np.pi/180 * np.array(
                      [ 150,
                       0,
                        130,
                        30,
                        150,
                        180
                       ]
                  ),
                  'speed_max': 0.3,   # maximum joint speed magnitude using speedj
                  'accel_max': 1,      # maximum acceleration magnitude of the leading axis using speedj
                  'reset_speed_limit': 0.5,
                  'q_ref': np.array([ 1.58724391, -2.6, 1.6, -0.71790582, 1.63685572, 1.00910473]),
                  'box_bound_buffer': 0.001,
                  'angle_bound_buffer': 0.001,
                  'ik_params':
                      (
                          0.089159, # d1
                          -0.42500, # a2
                          -0.39225, # a3
                          0.10915,  # d4
                          0.09465,  # d5
                          0.0823    # d6
                      )
              },
    'UR3_default':
              {
                  'host': '192.168.2.152',  # put UR3 Controller address here
                  'end_effector_low': np.array([-0.12, 0.2, 0.4]),
                  'end_effector_high': np.array([-0.12, 1.1, 0.9]),
                  'angles_low': np.array([-180, -120])*np.pi/180,
                  'angles_high': np.array([-90, 120])*np.pi/180,
                  'reset_speed_limit': 0.6,
                  'q_ref': np.array([1.49595487, -1.8, 1.19992781, -3.0167721, -1.54870445, 3.11713743]),
                  'box_bound_buffer': 0.001,
                  'angle_bound_buffer': 0.01,
                  'ik_params':
                      (
                          0.1273,   # d1
                          -0.612,    # a2
                          -0.5723,   # a3
                          0.163941, # d4
                          0.1157,   # d5
                          0.0922    # d6
                      )
              },
    'Visual-UR5':
              {
                  'host': '192.168.2.152',  # put UR5 Controller address here
                  #'end_effector_low': np.array([-0.3, -0.6, 0.5]),
                  #'end_effector_high': np.array([0.2, 0.4, 1.0]),
                  'end_effector_low': np.array([-0.4, 0.1, 0.1]),
                  'end_effector_high': np.array([0.4, 0.8, 0.8]),
                  'angles_low':np.pi/180 * np.array(
                      [ 55,
                       -120,#-180
                       -0,
                       -215,
                        -135,
                        50
                       ]
                  ),
                  'angles_high':np.pi/180 * np.array(
                      [ 125,
                       -30,
                        135,
                        -135,
                        -45,
                        190
                       ]
                  ),
                  'speed_max': 0.7,   # maximum joint speed magnitude using speedj
                  'accel_max': 1,      # maximum acceleration magnitude of the leading axis using speedj
                  'reset_speed_limit': 0.5,
                  #'q_ref': np.array([ 1.58724391, -2.0, 2.1, -3.01790582, -1.63685572, 3.1415926]),
                  #'q_ref': np.array([ 1.5707, -2.0, 1.9, -3.0415, -1.5707, 3.1415]),
                  'q_ref': np.array([ 1.5707, -2.0, 2.17, -3.0415, -1.5707, 3.1415]),
                  'box_bound_buffer': 0.001,
                  'angle_bound_buffer': 0.001,
                  'ik_params':
                      (
                          0.089159, # d1
                          -0.42500, # a2
                          -0.39225, # a3
                          0.10915,  # d4
                          0.09465,  # d5
                          0.0823    # d6
                      )
              },
    'Visual-UR5-min-time':
              {
                  'host': '192.168.2.152',  # put UR5 Controller address here
                  #'end_effector_low': np.array([-0.3, -0.6, 0.5]),
                  #'end_effector_high': np.array([0.2, 0.4, 1.0]),
                  #'end_effector_low': np.array([-0.4, 0.1, 0.1]),
                  #'end_effector_high': np.array([0.4, 0.8, 0.8]),
                  'end_effector_low': np.array([-0.25, 0.2, 0.35]),
                  # for origin: -0.1, 0.3, 0.6
                  'end_effector_high': np.array([0.15, 0.55, 0.7]),
                  'angles_low':np.pi/180 * np.array(
                      [ 60, # base
                       -120, # shoulder
                       -0, # elbow
                       -250, # wrist 1
                        -135, # wrist 2
                        50 # wrist 3, fixed
                       ]
                  ),
                  'angles_high':np.pi/180 * np.array(
                      [ 115, # base
                       -30, # shoulder
                        135, # elbow
                        -135, # wrist 1
                        -45, # wrist 2
                        190 # wrist 3, fixed
                       ]
                  ),
                  'speed_max': 0.7,   # maximum joint speed magnitude using speedj
                  'accel_max': 1,      # maximum acceleration magnitude of the leading axis using speedj
                  'reset_speed_limit': 0.5,
                  #'q_ref': np.array([ 1.58724391, -2.0, 2.1, -3.01790582, -1.63685572, 3.1415926]),
                  'q_ref': np.array([ 1.5707, -2.0, 2.17, -3.0415, -1.5707, 3.1415]),
                  'box_bound_buffer': 0.001,
                  'angle_bound_buffer': 0.001,
                  'ik_params':
                      (
                          0.089159, # d1
                          -0.42500, # a2
                          -0.39225, # a3
                          0.10915,  # d4
                          0.09465,  # d5
                          0.0823    # d6
                      )
              },
}