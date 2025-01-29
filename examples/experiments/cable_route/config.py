import numpy as np
from franka_env.envs.franka_env import DefaultEnvConfig
from experiments.config import DefaultTrainingConfig
import os
import jax
import jax.numpy as jnp

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
)
from franka_env.envs.relative_env import RelativeFrame
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.cable_route.wrapper import CableRouteEnv
from experiments.usb_pickup_insertion.wrapper import GripperPenaltyWrapper


class EnvConfig(DefaultEnvConfig):
    """Set the configuration for FrankaEnv."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "241122072130",
            "dim": (1280, 720),
            "exposure": 40000,
        },
        "wrist_1_classifier": {
            "serial_number": "241122072130",
            "dim": (1280, 720),
            "exposure": 40000,
        },
        # "wrist_2": "127122270572",
    }
    IMAGE_CROP = {
        "wrist_1_classifier": lambda img: img[225:-225],
        "front_classifier": lambda img: img[289:336, 245:301],
        "side_classifier": lambda img: img[195:241, 136:202],
        "front": lambda img: img[162:431, 92:580],
    }
    GENERIC_CAMERAS = {
        "front": {"id_name": "usb-Razer_Inc_Razer_Kiyo_X_01.00.00-video-index0"},
        "front_classifier": {
            "id_name": "usb-Razer_Inc_Razer_Kiyo_X_01.00.00-video-index0"
        },
        "side": {"id_name": "usb-046d_HD_Pro_Webcam_C920-video-index0"},
        "side_classifier": {"id_name": "usb-046d_HD_Pro_Webcam_C920-video-index0"},
    }
    TARGET_POSE = np.array(
        [
            0.460639895728905,
            -0.02439473272513422,
            0.006321125814908725,
            3.1331234,
            0.0182487,
            1.5824805,
        ]
    )
    RESET_POSE = TARGET_POSE + np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    REWARD_THRESHOLD: np.ndarray = np.zeros(6)
    APPLY_GRIPPER_PENALTY = False
    ACTION_SCALE = np.array([0.05, 0.3, 1])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.1
    RANDOM_RZ_RANGE = np.pi / 6
    MAX_EPISODE_LENGTH = 125
    ABS_POSE_LIMIT_LOW = np.array(
        [
            TARGET_POSE[0] - RANDOM_XY_RANGE,
            TARGET_POSE[1] - RANDOM_XY_RANGE,
            TARGET_POSE[2] - 0.001,
            TARGET_POSE[3] - 0.01,
            TARGET_POSE[4] - 0.01,
            TARGET_POSE[5] - RANDOM_RZ_RANGE,
        ]
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            TARGET_POSE[0] + RANDOM_XY_RANGE,
            TARGET_POSE[1] + RANDOM_XY_RANGE,
            TARGET_POSE[2] + 0.1,
            TARGET_POSE[3] + 0.01,
            TARGET_POSE[4] + 0.01,
            TARGET_POSE[5] + RANDOM_RZ_RANGE,
        ]
    )
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.005,
        "translational_clip_y": 0.005,
        "translational_clip_z": 0.005,
        "translational_clip_neg_x": 0.005,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.02,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 3000,
        "translational_damping": 89,
        "rotational_stiffness": 300,
        "rotational_damping": 9,
        "translational_Ki": 0.1,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.05,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.05,
        "rotational_Ki": 0.1,
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist_1", "front", "side"]
    classifier_keys = ["side_classifier", "front_classifier", "wrist_1_classifier"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = CableRouteEnv(
            fake_env=fake_env, save_video=save_video, config=EnvConfig()
        )
        # env = GripperCloseEnv(env)
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs, **kwargs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))

                height_ok = True
                if "info" in kwargs and "original_state_obs" in kwargs["info"]:
                    tcp_z = kwargs["info"]["original_state_obs"]["tcp_pose"][2]
                    height_ok = tcp_z < 0.032
                gripper_closed = obs["state"][0, 0] < 0.4
                reward = sigmoid(classifier(obs).item())
                print("reward", reward)
                reward_is_certain = reward > 0.95
                return int(reward_is_certain and gripper_closed and height_ok)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env
