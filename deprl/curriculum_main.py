"""Script used to train agents on a curriculum."""

import os
import traceback

import numpy as np

import torch

from deprl import custom_distributed
from deprl.utils import load_checkpoint, prepare_params
from deprl.vendor.tonic import logger


def train(
    config,
):
    """
    Trains an agent on environments from a curriculum.
    """
    tonic_conf = config["tonic"]

    # Run the header first, e.g. to load an ML framework.
    if "header" in tonic_conf:
        exec(tonic_conf["header"])

    # In case no env_args are passed via the config
    if "env_args" not in config or config["env_args"] is None:
        config["env_args"] = {}

    # Make a list of environment names.
    if "environments" not in tonic_conf or tonic_conf["environments"] is None:
        raise ValueError("No environments specified.")
    _environments = [
        env.strip() for env in tonic_conf["environments"].split(";")
    ]

    # get all of the environments, curriculum changes between them
    print(f"Environments: {_environments}")
    environments = []

    for _environment in _environments:
        environment = custom_distributed.distribute(
            environment=_environment,
            tonic_conf=tonic_conf,
            env_args=config["env_args"],
        )
        environment.initialize(seed=tonic_conf["seed"] + len(environments))
        environments.append(environment)

    # Build the testing environment.

    if (
        "test_environment" in tonic_conf
        and tonic_conf["test_environment"] is not None
    ):
        _test_environment = tonic_conf["test_environment"]
        test_env_args = (
            config["test_env_args"]
            if "test_env_args" in config
            else config["env_args"]
        )
        test_environment = custom_distributed.distribute(
            environment=_test_environment,
            tonic_conf=tonic_conf,
            env_args=test_env_args,
            parallel=1,
            sequential=1,
        )
        test_environment.initialize(seed=tonic_conf["seed"] + 1000000)

        test_environment.start()
    else:
        for _test_environment in _environments:
            test_environment = custom_distributed.distribute(
                environment=_environments[len(environments) - 1],
                tonic_conf=tonic_conf,
                env_args=config["env_args"],
                parallel=1,
                sequential=1,
            )
            test_environment.initialize(seed=tonic_conf["seed"] + 1000000)

            test_environment.start()

    # Build the agent.
    if "agent" not in tonic_conf or tonic_conf["agent"] is None:
        raise ValueError("No agent specified.")
    agent = eval(tonic_conf["agent"])

    # check if all environments have the same observation and action space
    for environment in environments:
        if environment.observation_space != environments[0].observation_space:
            raise ValueError("Environments have different observation spaces.")
        if environment.action_space != environments[0].action_space:
            raise ValueError("Environments have different action spaces.")

    # Set custom parameters
    if "model_args" in config:
        agent.set_custom_params(config["model_args"])
    agent.initialize(
        observation_space=environments[-1].observation_space,
        action_space=environments[-1].action_space,
        seed=tonic_conf["seed"],
    )

    # Set DEP parameters
    if hasattr(agent, "expl") and "DEP" in config:
        agent.set_dep_params(**config["DEP"])

    # Initialize the logger to get paths
    logger.initialize(
        script_path=__file__,
        config=config,
        test_env=test_environment,
        resume=tonic_conf["resume"],
    )
    path = logger.get_path()

    # Process the checkpoin path same way as in tonic_conf.play
    checkpoint_path = os.path.join(path, "checkpoint.pth")

    time_dict = {"steps": 0, "epochs": 0, "episodes": 0}
    (
        _,
        checkpoint_path,
        loaded_time_dict,
    ) = load_checkpoint(checkpoint_path, checkpoint="last")
    time_dict = time_dict if loaded_time_dict is None else loaded_time_dict

    if checkpoint_path:
        # Load the logger from a checkpoint.
        logger.load(checkpoint_path, time_dict)
        # Load the weights of the agent form a checkpoint.
        agent.load(checkpoint_path)

    # Build the trainer.
    trainer = tonic_conf["trainer"] or "tonic_conf.Trainer()"
    trainer = eval(trainer)
    trainer.initialize(
        agent=agent,
        environments=environments,
        test_environment=test_environment,
        full_save=tonic_conf["full_save"],
    )

    # Run some code before training.
    if tonic_conf["before_training"]:
        exec(tonic_conf["before_training"])

    # Train.
    try:
        trainer.run(config, **time_dict)
    except Exception as e:
        logger.log(f"trainer failed. Exception: {e}")
        traceback.print_tb(e.__traceback__)

    # Run some code after training.
    # if tonic_conf["after_training"]:
    #     exec(["after_training"])


def set_tensor_device():
    # use CUDA or apple metal
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        logger.log("CUDA detected, storing default tensors on it.")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")
        logger.log("MPS detected, storing default tensors on it.")
    else:
        logger.log("No CUDA or MPS detected, running on CPU")


def main():
    config = prepare_params()
    if "cpu_override" in config["tonic"] and config["tonic"]["cpu_override"]:
        torch.set_default_device("cpu")
        logger.log("Manually forcing CPU run.")
    else:
        set_tensor_device()
    print(config)
    train(config)


if __name__ == "__main__":
    main()
