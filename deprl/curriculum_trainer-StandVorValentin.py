import os
import time

import numpy as np
import torch

from collections import deque

from deprl.custom_test_environment import (
    test_dm_control,
    test_mujoco,
    test_scone,
    #test_scone_vel,
)
from deprl.vendor.tonic import logger

if "ROBOHIVE_VERBOSITY" not in os.environ:
    os.environ["ROBOHIVE_VERBOSITY"] = "ALWAYS"


class Trainer:
    """Trainer used to train and evaluate an agent on an environment."""

    def __init__(
        self,
        steps=1e7,
        epoch_steps=2e4,
        save_steps=5e5,
        test_episodes=20,
        show_progress=True,
        replace_checkpoint=False,
    ):
        assert epoch_steps <= save_steps
        self.max_steps = int(steps)
        self.epoch_steps = int(epoch_steps)
        self.save_steps = int(save_steps)
        self.test_episodes = test_episodes
        self.show_progress = show_progress
        self.replace_checkpoint = replace_checkpoint

    def initialize(
        self, agent, environments, test_environment=None, full_save=False
    ):
        self.full_save = full_save
        self.agent = agent
        self.environments = environments
        self.test_environment = test_environment
        self.number_of_environments = len(environments)

        self.environment = None

    def run(self, params, steps=0, epochs=0, episodes=0):
        """Runs the main training loop."""

        start_time = last_epoch_time = time.time()

        # Start the environments.
        observations_list = []
        muscle_states_list = []

        for environment in self.environments:
            observations, muscle_states = environment.start()
            observations_list.append(observations)
            muscle_states_list.append(muscle_states)

        num_workers = len(observations)
        scores = np.zeros(num_workers)
        lengths = np.zeros(num_workers, int)
        unique_states = [set() for _ in range(num_workers)]
        self.steps, epoch_steps = steps, 0
        steps_since_save = 0

        # keep track of data that can be used to update the curriculum
        length_percentages = []
        velocities = []
        angles = []

        # get the initial curriculum parameters
        environment_turn = self.agent.replay.last_env_index
        angle_range = self.agent.replay.last_angle_range
        vel_range = self.agent.replay.last_vel_range
        stand_prob = self.agent.replay.last_stand_prob
        task = self.agent.replay.last_task

        while True:

            self.environment = self.environments[environment_turn]
            observations = observations_list[environment_turn]
            muscle_states = muscle_states_list[environment_turn]

            current_velocities = self.environment.get_vel()
            #test_scone_vel(self.test_environment, self.agent, steps, params)
            #self.environment.get_vel()
            current_angles = self.environment.get_angles()
            reward_scale=self.environment.get_reward_scale()

            # print('reward_scale curr trainer:', reward_scale)

            velocities.extend(current_velocities)
            angles.extend(current_angles)

            # Select actions.
            if hasattr(self.agent, "expl"):
                greedy_episode = (
                    not episodes % self.agent.expl.test_episode_every
                )
            else:
                greedy_episode = None
            assert not np.isnan(observations.sum())
            actions = self.agent.step(
                observations, self.steps, muscle_states, greedy_episode
            )
            assert not np.isnan(actions.sum())
            # raise Exception(f'{type(self.environment.environments[0])}')
            logger.store("train/action", actions, stats=True)

            # action variance is calculated as the mean of the variances of each column (muscle activation of every agent action)
            action_variance = np.var(actions, axis=0)
            # store the mean of the action variance
            logger.store(
                "train/action_variance", action_variance.mean(), stats=True
            )

            for i, obs in enumerate(observations):
                unique_states[i].add(tuple(obs.flatten()))

            # Take a step in the environments.
            observations, muscle_states, info = self.environment.step(
                actions, angle_range, vel_range, stand_prob, task
            )
            observations_list[environment_turn] = observations
            muscle_states_list[environment_turn] = muscle_states

            if "env_infos" in info:
                info.pop("env_infos")
            self.agent.update(**info, steps=self.steps)

            scores += info["rewards"]
            lengths += 1
            self.steps += num_workers
            epoch_steps += num_workers
            steps_since_save += num_workers

            # Show the progress bar.
            if self.show_progress:
                logger.show_progress(
                    self.steps, self.epoch_steps, self.max_steps
                )

            # Check the finished episodes.
            for i in range(num_workers):
                if info["resets"][i]:
                    logger.store("train/episode_score", scores[i], stats=True)
                    logger.store(
                        "train/episode_length", lengths[i], stats=True
                    )
                    # store the current environment parameters
                    logger.store("train/environment_index", environment_turn)
                    logger.store("train/angle_range", angle_range[1])
                    logger.store("train/vel_range", vel_range[1])
                    logger.store("train/stand_prob", stand_prob)
                    logger.store("train/task", task)

                    if i == 0:
                        # adaptive energy cost
                        if hasattr(self.agent.replay, "action_cost"):
                            logger.store(
                                "train/action_cost_coeff",
                                self.agent.replay.action_cost,
                            )
                            self.agent.replay.adjust(scores[i])

                    length_percentages.append(
                        lengths[i] / self.environment._max_episode_steps
                    )
                    scores[i] = 0
                    lengths[i] = 0
                    episodes += 1

                    logger.store("train/unique_states", len(unique_states[i]))

                    unique_states[i] = set()

            # End of the epoch.
            if epoch_steps >= self.epoch_steps:
                # Evaluate the agent on the test environment.
                if self.test_environment is not None:
                    if (
                        "control"
                        in str(
                            type(
                                self.test_environment.environments[0].unwrapped
                            )
                        ).lower()
                    ):
                        _ = test_dm_control(
                            self.test_environment, self.agent, steps, params
                        )

                    elif (
                        "scone"
                        in str(
                            type(
                                self.test_environment.environments[0].unwrapped
                            )
                        ).lower()
                    ):
                        _ = test_scone(
                            self.test_environment, self.agent, steps, params
                        )

                    else:
                        _ = test_mujoco(
                            self.test_environment, self.agent, steps, params
                        )
                # print('test_scone', test_scone(self.test_environment, self.agent, steps, params))

                # Log the data.
                epochs += 1
                current_time = time.time()
                epoch_time = current_time - last_epoch_time
                sps = epoch_steps / epoch_time
                logger.store("train/episodes", episodes)
                logger.store("train/epochs", epochs)
                logger.store("train/seconds", current_time - start_time)
                logger.store("train/epoch_seconds", epoch_time)
                logger.store("train/epoch_steps", epoch_steps)
                logger.store("train/steps", self.steps)
                logger.store("train/worker_steps", self.steps // num_workers)
                logger.store("train/steps_per_second", sps)
                last_epoch_time = time.time()
                epoch_steps = 0

                logger.dump()

                # update the curriculum, once per epoch
                environment_turn, angle_range, vel_range, stand_prob, task = (
                    self.agent.replay._curriculum_step(
                        num_envs=self.number_of_environments,
                        length_percentages=length_percentages,
                        velocities=velocities,
                        angles=angles,
                        steps_per=self.steps / self.max_steps,
                        reward_scale=reward_scale,
                    )
                )

                velocities = []
                angles = []
                length_percentages = []

            # End of training.
            stop_training = self.steps >= self.max_steps

            # Save a checkpoint.
            if stop_training or steps_since_save >= self.save_steps:
                path = os.path.join(logger.get_path(), "checkpoints")
                if os.path.isdir(path) and self.replace_checkpoint:
                    for file in os.listdir(path):
                        if file.startswith("step_"):
                            os.remove(os.path.join(path, file))
                checkpoint_name = f"step_{self.steps}"
                save_path = os.path.join(path, checkpoint_name)
                # save agent checkpoint
                self.agent.save(save_path, full_save=self.full_save)
                # save logger checkpoint
                logger.save(save_path)
                # save time iteration dict
                self.save_time(save_path, epochs, episodes)
                steps_since_save = self.steps % self.save_steps
                current_time = time.time()

            if stop_training:
                self.close_mp_envs()
                return scores

    def close_mp_envs(self):
        for environment in self.environments:
            for index in range(len(environment.processes)):
                environment.processes[index].terminate()
                environment.action_pipes[index].close()
            environment.output_queue.close()

    def save_time(self, path, epochs, episodes):
        time_path = self.get_path(path, "time")
        time_dict = {
            "epochs": epochs,
            "episodes": episodes,
            "steps": self.steps,
        }
        torch.save(time_dict, time_path)

    def get_path(self, path, post_fix):
        return path.split("step")[0] + post_fix + ".pt"
