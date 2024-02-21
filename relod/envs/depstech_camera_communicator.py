# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import cv2 as cv
import numpy as np

from senseact.communicator import Communicator


class CameraCommunicator(Communicator):
    """
    Camera Communicator for interfacing with most common webcams supported by OpenCV.
    """

    def __init__(self, res=(0, 0), device_id=0):
        """Inits the camera communicator with desired resolution and device_id.

        Args:
            res: a 2D tuple representing desired frame resolution. A default
                resolution is used if res contains zero values.
            device_id: either the device_id as integer or as string to the
                path of the device
        """
        self._device_id = device_id

        # query camera default resolution or test custom resolution if needed and release it so that
        # the camera can be opened in the child process
        temp_cap = cv.VideoCapture(device_id, cv.CAP_V4L2)

        if not temp_cap.isOpened():
            raise IOError("Unable to open camera on device id {}".format(self._device_id))

        if res[0] == 0 or res[1] == 0:
            # get the default resolution, and extract depth from OpenCV typecode such as CV_8UC1
            res = (temp_cap.get(cv.CAP_PROP_FRAME_WIDTH),
                   temp_cap.get(cv.CAP_PROP_FRAME_HEIGHT),
                   1 + (int(temp_cap.get(cv.CAP_PROP_FORMAT)) >> 3))
        else:
            temp_cap.set(cv.CAP_PROP_FRAME_WIDTH, res[0])
            temp_cap.set(cv.CAP_PROP_FRAME_HEIGHT, res[1])
            real_res = (temp_cap.get(cv.CAP_PROP_FRAME_WIDTH),
                        temp_cap.get(cv.CAP_PROP_FRAME_HEIGHT),
                        1 + (int(temp_cap.get(cv.CAP_PROP_FORMAT)) >> 3))

            if real_res[0] != res[0] or real_res[1] != res[1]:
                print("Custom resolution ({}, {}) not supported, reset to default and use resize.".format(res[0], res[1]))

            self._real_res = real_res

        temp_cap.release()

        self._res = res
        self._cap = None

        sensor_args = {'array_len': int(3*np.product(self._res)),
                       'array_type': 'B',
                       'np_array_type': 'B',
                       }

        super(CameraCommunicator, self).__init__(use_sensor=True,
                                                 use_actuator=False,
                                                 sensor_args=sensor_args,
                                                 actuator_args={})

    def run(self):
        """Opening the video IO in the child process and invoke parent 'run' """
        self._cap = cv.VideoCapture(self._device_id, cv.CAP_V4L2)

        if not self._cap.isOpened():
            raise IOError("Unable to open camera on device id {}".format(self._device_id))

        self._cap.set(cv.CAP_PROP_FRAME_WIDTH, self._real_res[0])
        self._cap.set(cv.CAP_PROP_FRAME_HEIGHT, self._real_res[1])
        self._cap.set(cv.CAP_PROP_FPS, 30)

        # main process loop
        super(CameraCommunicator, self).run()

        # try to close the IO when the process end
        self._cap.release()

    def _sensor_handler(self):
        """Block and read the next available frame."""
        # reading the original frame in (height, width, depth) dimension
        retval, frame = self._cap.read()
        if retval:
            frame = cv.resize(frame, self._res)
            # flatten and write to buffer
            self.sensor_buffer.write(frame.flatten())

    def _actuator_handler(self):
        """There's no actuator available for cameras."""
        raise RuntimeError("Camera Communicator does not have an actuator handler.")

    def get_image(self):
        return self.sensor_buffer.read_update()