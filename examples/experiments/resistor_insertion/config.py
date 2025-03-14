import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.resistor_insertion.wrapper import ResistorEnv, GripperPenaltyWrapper


class EnvConfig(DefaultEnvConfig):
    # SERVER_URL = "http://127.0.0.1:5000/"
    SERVER_URL = "http://0.0.0.0:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "241122072130",
            "dim": (1280, 720),
            "exposure": 10500,
        },
        "wrist_2": {
            "serial_number": "313522071070",
            "dim": (1280, 720),
            "exposure": 10500,
        },
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[:, 365:1085],
        "wrist_2": lambda img: img[:, 365:1085],
        "side": lambda img: img[160:325, 105:270]
    }
    GENERIC_CAMERAS = {
        #"side": {"id_name": "usb-Microsoft_Azure_Kinect_4K_Camera_000471215012-video-index0"},
        "side": {"id_name": "usb-USB2.0_Camera_USB2.0_Camera-video-index0"},
        # "side": {"id_name": "usb-046d_HD_Pro_Webcam_C920-video-index0"},
    }
    TARGET_POSE = np.array(
        # [0.5043032062790537,0.22067800416581693,0.06040979312278208,np.pi,0.0026128038950117283,1.654021177685403]
        [0.6182235085854079,0.2222460209702428,0.04398205916057833, np.pi, 0,0]
        # [0.4881879113354628,0.2208223800177667,0.05731334740852401, np.pi, 0, np.pi/2]
    )
    GRASP_POSE = np.array(
        # [0.634300840464846,0.17992817965194208,0.06180436925694141, np.pi, 0, np.pi/2]
        # [0.6797256032879704,0.148140473338448678,0.060789280023034964, np.pi, 0, np.pi/2] # good one
        # [0.6797256032879704,0.135140473338448678,0.055789280023034964, np.pi, 0, np.pi/2]
        [0.6082500302779233,0.15106681669594402,0.02541610749323761,np.pi, 0,0]
    )
    RESET_POSE = TARGET_POSE + np.array([0, 0, 0.02, 0, 0.0, 0])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.04, 0.02, 0.03, np.pi/12, np.pi/12, np.pi/12])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.04, 0.02, 0.03, np.pi/12, np.pi/12, np.pi/12])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.1
    ACTION_SCALE = np.array([0.025, 0.1, 1])
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 200
    COMPLIANCE_PARAM = {
        "translational_stiffness": 1500,
        "translational_damping": 89,
        "rotational_stiffness": 120,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.005,
        "translational_clip_y": 0.005,
        "translational_clip_z": 0.0075,
        "translational_clip_neg_x": 0.005,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.0075,
        "rotational_clip_x": 0.025,
        "rotational_clip_y": 0.025,
        "rotational_clip_z": 0.025,
        "rotational_clip_neg_x": 0.025,
        "rotational_clip_neg_y": 0.025,
        "rotational_clip_neg_z": 0.025,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.03,
        "rotational_clip_y": 0.03,
        "rotational_clip_z": 0.1,
        "rotational_clip_neg_x": 0.03,
        "rotational_clip_neg_y": 0.03,
        "rotational_clip_neg_z": 0.1,
        "rotational_Ki": 0.0,
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist_1", "wrist_2", "side"]
    classifier_keys = ["wrist_1", "wrist_2", "side"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = ResistorEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
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

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                prediction = sigmoid(classifier(obs))
                print("Predict: ", prediction)
                # added check for z position to further robustify classifier, but should work without as well
                success = int((prediction > 0.95).item())
                # if success:
                #     input("Success")
                return success

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env
