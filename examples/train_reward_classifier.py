import glob
import os
import pickle as pkl
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
import optax
from tqdm import tqdm
from absl import app, flags
# import cv2
import numpy as np
from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier

from experiments.mappings import CONFIG_MAPPING
import datetime


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 150, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)

    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)

    # Create buffer for positive transitions
    pos_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=20000,
        include_label=True,
    )

    # merged_successes = []
    success_data = []
    success_paths = glob.glob(
        os.path.join(os.getcwd(), "classifier_data", "*success*.pkl")
    )
    for path in success_paths:
        success_data += pkl.load(open(path, "rb"))
        # merged_successes += success_data
    np.random.shuffle(success_data)

    for i, trans in enumerate(success_data):
        if "images" in trans["observations"].keys():
            continue
        trans["labels"] = 1
        trans["actions"] = env.action_space.sample()
        pos_buffer.insert(trans)
        # frame = np.concatenate(
        #     [
        #         trans["observations"]["wrist_1"][0],
        #         trans["observations"]["wrist_2"][0],
        #         trans["observations"]["side"][0]
        #     ],
        #     axis=1,
        # )
        # cv2.imshow("concat", frame)
        # cv2.waitKey(1)
        # print(f"Success img. {i}/{len(success_data)}")
    # with open(os.path.join(os.getcwd(), "classifier_data", f"merged_success_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"), "wb") as f:
    #     np.random.shuffle(merged_successes)
    #     pkl.dump(merged_successes, f)

    pos_iterator = pos_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )

    # Create buffer for negative transitions
    neg_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=50000,
        include_label=True,
    )

    # merged_failures = []
    failure_data = []
    failure_paths = glob.glob(
        os.path.join(os.getcwd(), "classifier_data", "*failure*.pkl")
    )
    for path in failure_paths:
        failure_data += pkl.load(open(path, "rb"))
    np.random.shuffle(failure_data)

        # merged_failures += failure_data
    for i, trans in enumerate(failure_data):
        if "images" in trans["observations"].keys():
            continue
        trans["labels"] = 0
        trans["actions"] = env.action_space.sample()
        neg_buffer.insert(trans)
        # frame = np.concatenate(
        #     [
        #         trans["observations"]["wrist_1"][0],
        #         trans["observations"]["wrist_2"][0],
        #         trans["observations"]["side"][0]
        #     ],
        #     axis=1,
        # )
        # cv2.imshow("concat", frame)
        # cv2.waitKey(1)
        # print(f"Failure img. {i}/{len(failure_data)}")
    # with open(os.path.join(os.getcwd(), "classifier_data", f"merged_failure_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"), "wb") as f:
    #     np.random.shuffle(merged_failures)
    #     pkl.dump(merged_failures, f)

    neg_iterator = neg_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )

    print(f"failed buffer size: {len(neg_buffer)}")
    print(f"success buffer size: {len(pos_buffer)}")

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)

    rng, key = jax.random.split(rng)
    classifier = create_classifier(
        key,
        sample["observations"],
        config.classifier_keys,
    )

    def data_augmentation_fn(rng, observations):
        for pixel_key in config.classifier_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    @jax.jit
    def train_step(state, batch, key):
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                batch["observations"],
                rngs={"dropout": key},
                train=True,
            )
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        logits = state.apply_fn(
            {"params": state.params},
            batch["observations"],
            train=False,
            rngs={"dropout": key},
        )
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])

        return state.apply_gradients(grads=grads), loss, train_accuracy

    for epoch in tqdm(range(FLAGS.num_epochs)):
        # Sample equal number of positive and negative examples
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        # Merge and create labels
        batch = concat_batches(pos_sample, neg_sample, axis=0)
        rng, key = jax.random.split(rng)
        obs = data_augmentation_fn(key, batch["observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "labels": batch["labels"][..., None],
            }
        )

        rng, key = jax.random.split(rng)
        classifier, train_loss, train_accuracy = train_step(classifier, batch, key)

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )

    checkpoints.save_checkpoint(
        os.path.join(os.getcwd(), "classifier_ckpt/"),
        classifier,
        step=FLAGS.num_epochs,
        overwrite=True,
    )


if __name__ == "__main__":
    app.run(main)
