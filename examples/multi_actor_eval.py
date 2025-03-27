#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_list("exp_names", None, "Name of experiments corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_list("checkpoint_paths", None, "Path to save checkpoints.")
flags.DEFINE_list("eval_checkpoint_steps", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################

def multi_actor_eval(agents, envs, sampling_rngs):

    success_counter = 0
    time_list = []

    for episode in range(FLAGS.eval_n_trajs):
        for i in range(len(agents)):
            env = envs[i]
            env.open_threads()
            # env = configs[i].get_environment(
            #     fake_env=False,
            #     save_video=False,
            #     classifier=True,
            # )
            # env = RecordEpisodeStatistics(env)
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rngs[i])
                actions = agents[i].sample_actions(
                    observations=jax.device_put(obs), argmax=False, seed=key
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs

                if done:
                    break

            if reward:
                dt = time.time() - start_time
                time_list.append(dt)
                print(dt)
                env.define_should_regrasp(False)
            elif i < len(agents) - 1:
                env.close()
                for env_ in envs:
                    env_.define_should_regrasp(True)
                # del env
                break
            if i == len(agents) - 1:
                success_counter += reward
                print(reward)
                print(f"{success_counter}/{episode + 1}")
                for env_ in envs:
                    env_.define_should_regrasp(True)
            # env.reset()
            env.close()
            # del env

    print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
    print(f"average time: {np.mean(time_list)}")
    return  # after done eval, return and exit

##############################################################################

def main(_):
    agents = []
    envs = []
    # configs = []
    sampling_rngs = []

    for i in range(len(FLAGS.exp_names)):
        config = CONFIG_MAPPING[FLAGS.exp_names[i]]()

        assert config.batch_size % num_devices == 0
        # seed
        rng = jax.random.PRNGKey(FLAGS.seed)
        rng, sampling_rng = jax.random.split(rng)

        assert FLAGS.exp_names[i] in CONFIG_MAPPING, "Experiment folder not found."
        env = config.get_environment(
            fake_env=False,
            save_video=False,
            classifier=True,
            open_threads=False
        )
        # env = RecordEpisodeStatistics(env)

        rng, sampling_rng = jax.random.split(rng)

        if (
            config.setup_mode == "single-arm-fixed-gripper"
            or config.setup_mode == "dual-arm-fixed-gripper"
        ):
            agent: SACAgent = make_sac_pixel_agent(
                seed=FLAGS.seed,
                sample_obs=env.observation_space.sample(),
                sample_action=env.action_space.sample(),
                image_keys=config.image_keys,
                encoder_type=config.encoder_type,
                discount=config.discount,
            )
        elif config.setup_mode == "single-arm-learned-gripper":
            agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
                seed=FLAGS.seed,
                sample_obs=env.observation_space.sample(),
                sample_action=env.action_space.sample(),
                image_keys=config.image_keys,
                encoder_type=config.encoder_type,
                discount=config.discount,
            )
        elif config.setup_mode == "dual-arm-learned-gripper":
            agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
                seed=FLAGS.seed,
                sample_obs=env.observation_space.sample(),
                sample_action=env.action_space.sample(),
                image_keys=config.image_keys,
                encoder_type=config.encoder_type,
                discount=config.discount,
            )
        else:
            raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

        # replicate agent across devices
        # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
        agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

        if FLAGS.checkpoint_paths[i] is not None and os.path.exists(FLAGS.checkpoint_paths[i]):
            # input("Checkpoint path already exists. Press Enter to resume training.")
            ckpt = checkpoints.restore_checkpoint(
                os.path.abspath(FLAGS.checkpoint_paths[i]),
                agent.state,
                step=int(FLAGS.eval_checkpoint_steps[i])
            )
            agent = agent.replace(state=ckpt)
            

        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())


        agents.append(agent)
        envs.append(env)
        # configs.append(config)
        sampling_rngs.append(sampling_rng)

        # env.close()
        # del env

    # actor loop
    print_green("starting actor loop")
    multi_actor_eval(
        agents,
        envs,
        sampling_rngs,
    )


if __name__ == "__main__":
    app.run(main)
