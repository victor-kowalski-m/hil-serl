import copy
import time
from franka_env.utils.rotations import euler_2_quat
import numpy as np
import requests
from pynput import keyboard
import gymnasium as gym
from franka_env.envs.franka_env import FrankaEnv


class ResistorEnv(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.should_regrasp = True

        # def on_press(key):
        #     if str(key) == "Key.f1":
        #         self.should_regrasp = True

        # listener = keyboard.Listener(on_press=on_press)
        # listener.start()

    def go_to_reset(self, joint_reset=False):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """
        # use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)

        # pull up
        # self._update_currpos()
        # reset_pose = copy.deepcopy(self.currpos)
        # reset_pose[2] = self.resetpos[2] + 0.04
        # self.interpolate_move(reset_pose, timeout=1)

        # perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        # perform Cartesian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self._send_pos_command(reset_pose)
        else:
            reset_pose = self.resetpos.copy()
            self._send_pos_command(reset_pose)
        time.sleep(0.5)

        # Change to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

    def regrasp(self):
        
        top_pose = self.config.GRASP_POSE.copy()
        top_pose[2] += 0.05
        # top_pose[0] += np.random.uniform(-0.005, 0.005)
        self.interpolate_move(top_pose, timeout=1)
        time.sleep(0.5)

        # input("Grasp?")
        grasp_pose = top_pose.copy()
        grasp_pose[2] -= 0.05
        self.interpolate_move(grasp_pose, timeout=0.5)

        requests.post(self.url + "close_gripper")
        self.last_gripper_act = time.time()
        time.sleep(2)

        self.interpolate_move(top_pose, timeout=0.5)
        time.sleep(0.2)

        self.interpolate_move(self.config.RESET_POSE, timeout=1)
        time.sleep(0.5)

    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording()

        # use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)

        # pull up
        self._update_currpos()
        # reset_pose = copy.deepcopy(self.currpos)
        # reset_pose[2] = self.resetpos[2] + 0.04
        # self.interpolate_move(reset_pose, timeout=1)

        # input("Press enter to release gripper...")
        time.sleep(0.5)
        self._send_gripper_command(1.0)
        time.sleep(1.0)
        # input("Place RAM in holder and press enter to grasp...")
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] = self.resetpos[2] + 0.05
        self.interpolate_move(reset_pose, timeout=1)
        time.sleep(1)

        # if True:
        if self.should_regrasp:
            self.regrasp()
            # self.should_regrasp = False

        self._recover()
        self.go_to_reset(joint_reset=False)
        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        for i in range(10):
            obs = self._get_obs()
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        self.terminate = False
        return obs, {}

class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info