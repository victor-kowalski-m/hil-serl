import copy
import os
import re
import time
from typing import OrderedDict

from cv2 import VideoCapture as CVVideoCapture
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from franka_env.envs.franka_env import FrankaEnv

# from examples.experiments.async_cable_route_drq.config import EnvConfig

##############################################################################


class CableRouteEnv(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def go_to_reset(self, joint_reset=False):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """
        self._send_gripper_command(1, force=True)
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.5)

        # Move up to clear the slot
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] += 0.05
        self.interpolate_move(reset_pose, timeout=1)

        # execute the go_to_rest method from the parent class
        super().go_to_reset(joint_reset)

    def init_realsense_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            if cam_name == "wrist_1_classifier":
                self.cap["wrist_1_classifier"] = self.cap["wrist_1"]
                continue
            cap = VideoCapture(RSCapture(name=cam_name, **kwargs))
            self.cap[cam_name] = cap

    def init_generic_cameras(self, cam_dict):
        def get_camera_by_id_name(cam_name):
            cam_file = "/dev/v4l/by-id/" + cam_name
            if not os.path.exists(cam_file):
                raise ValueError(f"Camera {cam_name} does not exist. Try reconnecting")

            device_path = os.path.realpath(cam_file)
            device_re = re.compile("\/dev\/video(\d+)")
            info = device_re.match(device_path)
            if info:
                device_num = int(info.group(1))
                return device_num
            else:
                raise RuntimeError("/dev/videoX not found. Not sure what to do.")

        for cam_name, val in cam_dict.items():
            if cam_name == "side_classifier":
                self.cap["side_classifier"] = self.cap["side"]
                continue
            elif cam_name == "front_classifier":
                self.cap["front_classifier"] = self.cap["front"]
                continue
            if "id_name" in val:
                name = get_camera_by_id_name(val["id_name"])
            else:
                name = val["name"]
            cap = CVVideoCapture(name)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera {name}")
            self.cap[cam_name] = cap
