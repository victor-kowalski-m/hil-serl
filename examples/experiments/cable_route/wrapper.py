import time
import copy

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
        self._send_gripper_command(-1, force=True)
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
