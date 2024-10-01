import numpy as np

from deprl.vendor.tonic import logger
from deprl.vendor.tonic.replays import Buffer


class CurriculumBuffer(Buffer):
    """
    Assume all activity is appended at the end of the observation.
    True for Myosuite and scone so far.
    """

    def __init__(self, *args, **kwargs):
        # performance threshold that needs to be achieved
        # still needs to be implemented
        self.threshold = kwargs.pop("threshold", 1000)

        # after switching the environment, the agent cannot switch again for a certain number of steps
        self.no_switch = 0

        # initialize the environment index, angle range, velocity range, and standing probability
        self.last_env_index = 0
        self.last_angle_range = (0, 0)
        self.last_vel_range = (1, 1)
        self.last_stand_prob = 0
        # 0 - target task, 1 - velocity task, 2 - orientation task
        # initial task is for the 4 year old
        self.last_task = 1

        # get the mode for switching the environment and the targets (angle, velocity, standing)
        self.mode_env = kwargs.pop("mode_env", 0)
        self.mode_target = kwargs.pop("mode_target", 0)

        if self.mode_env not in [0, 1]:
            raise Exception(
                f"Mode {self.mode_env} of the environment is not implemented."
            )
        if self.mode_target not in [0, 1]:
            raise Exception(
                f"Mode {self.mode_target} of the targets is not implemented."
            )

        # Initial values ----------------
        self.cdt_avg = 0
        self.score_avg = 0
        super().__init__(*args, **kwargs)

    def _curriculum_step(
        self,
        num_envs=2,
        velocities=None,
        length_percentages=None,
        angles=None,
        steps_per=0,
    ):
        """Perform a curriculum step. Update the environment index, angle range, and velocity range.

        Args:
            num_envs (int): The number of environments in the curriculum.
            velocities (list): The list of the tuples of the current and target velocities of the tasks.
            length_percentages (list): The lists of the length_percentages of the tasks.
            angles (list): The list of the tuples of the current and target angles of the tasks.

        Returns:
            int: The index of the environment to use.
            list: The new angle range.
            list: The new velocity range.
            float: The probability of tbe standing task to be selected.
            int: The task to be selected.
        """

        old_env_index = self.last_env_index

        if length_percentages is None:
            raise Exception(
                "length_percentage cannot be None to perform a curriculum step."
            )
        if angles is None:
            raise Exception(
                "angles cannot be None to perform a curriculum step."
            )
        if velocities is None:
            raise Exception(
                "velocities cannot be None to perform a curriculum step."
            )

        env_0_threshold = [0.35, 0.2]
        env_1_threshold = 0.2

        if self.no_switch > 0:
            self.no_switch -= 1

        elif self.mode_env == 0:
            # change the environment based on the average length percentage of an episode
            avg_length_percentage = np.mean(length_percentages)
            if self.last_env_index == 0:
                if avg_length_percentage >= env_0_threshold[0]:
                    self.last_env_index = 1
            elif self.last_env_index == 1:
                if avg_length_percentage <= env_0_threshold[1]:
                    self.last_env_index = 0

        elif self.mode_env == 1:
            # change the environments at a fixed percentage of steps
            self.last_task = 2
            if steps_per >= env_1_threshold:
                self.last_env_index = 1
                self.last_task = 1

        # if the environment was switched, the agent cannot switch again for a certain number of steps
        if self.last_env_index != old_env_index:
            self.no_switch = 5

        target_0_threshold = [0.6, 0.15]
        # target_1_threshold[0] = env_1_threshold, so that the ranges are not changed until the environment is switched to the adult
        target_1_threshold = [0.2, 0.4, 0.6]

        if self.mode_target == 0:
            # increase the velocity range if the average difference between the current and target velocities is below a threshold
            vel_percent_diffs = [
                abs(velocity[0] - velocity[1]) / (velocity[1] + 0.0001)
                for velocity in velocities
            ]
            # increase the angle range if the average difference between the current and target angles is below a threshold
            angle_percent_diffs = [
                abs(angle[0] - angle[1]) / np.pi for angle in angles
            ]

            # only take the lowest 3/4 of the percentages, to avoid initial outliers
            vel_percent_diff = np.mean(
                vel_percent_diffs[: int(len(vel_percent_diffs) * 3 / 4)]
            )
            angle_percent_diff = np.mean(
                angle_percent_diffs[: int(len(angle_percent_diffs) * 3 / 4)]
            )

            if vel_percent_diff <= target_0_threshold[0]:
                vel_percent = min(1.25, self.last_vel_range[1] + 0.05)
                self.last_vel_range = (0.25, vel_percent)

            if angle_percent_diff <= target_0_threshold[1]:
                angle_percent = min(
                    np.pi, self.last_angle_range[1] + np.pi / 32
                )
                self.last_angle_range = (-angle_percent, angle_percent)

        elif self.mode_target == 1:
            # increase the ranges based on the number of steps, but not simultaneously
            if steps_per >= target_1_threshold[0]:
                vel_percent = min(1, (steps_per - target_1_threshold[0]) / 0.2)
                self.last_vel_range = (
                    1 - 0.75 * vel_percent,
                    1 + 0.25 * vel_percent,
                )
                # while increasing the velocity range, the velocity task is selected
                self.last_task = 1

            if steps_per >= target_1_threshold[1]:
                angle_percent = min(
                    1, (steps_per - target_1_threshold[1]) / 0.2
                )
                self.last_angle_range = (
                    -angle_percent * np.pi,
                    angle_percent * np.pi,
                )
                # while increasing the angle range, the orientation task is selected
                self.last_task = 2

            if steps_per >= target_1_threshold[2]:
                self.last_stand_prob = 1 - 0.95 * min(
                    1, (steps_per - target_1_threshold[2]) / 0.2
                )
                # while changing the standing probability and in the last phase, the combined task is selected
                self.last_task = 0

        return (
            self.last_env_index,
            self.last_angle_range,
            self.last_vel_range,
            self.last_stand_prob,
            self.last_task,
        )

    # not used in the current implementation, was used just for changing the environment
    def _get_env_index(
        self,
        num_envs=3,
        velocities=None,
        length_percentages=None,
        angles=None,
        steps_per=0,
    ):
        """Get the environment index for making the next step.

        Args:
            mode (int): The mode of the curriculum.
                    0: choose a random environment
                    1: choose the environment based on the velocity of the last environment
                    2: choose the environment based on the average length percentage of the last environment

            num_envs (int): The number of environments in the curriculum.
            velocities (list): The list of the tuples of the current and target velocities of the tasks.
            length_percentages (list): The lists of the length_percentages of the tasks.
            angles (list): The list of the tuples of the current and target angles of the tasks.

        Returns:
            int: The index of the environment to use.
        """

        modes = [0, 1, 2, 3, 4, 5]
        if self.mode_env not in modes:
            raise Exception(f"Mode {self.mode} not implemented.")

        old_env_index = self.last_env_index

        if self.no_switch > 0:
            self.no_switch -= 1

        elif self.mode_env == 0:
            self.last_env_index = np.random.randint(num_envs)

        elif self.mode_env == 1:

            # calculate the absolute percentage difference between the current and target velocities
            percent_diffs = [
                abs(velocity[0] - velocity[1]) / velocity[1]
                for velocity in velocities
            ]
            # take the mean of the lowest 3/4 of the percentages
            percent_diff = np.mean(
                sorted(percent_diffs)[: int(len(percent_diffs) * 3 / 4)]
            )

            # with open("velocities.txt", "a") as f:
            #     f.write(str(percent_diff) + "\n")

            if self.last_env_index == 0:
                if percent_diff <= 0.65:
                    self.last_env_index = 1
            elif self.last_env_index == 1:
                if percent_diff >= 0.8:
                    self.last_env_index = 0
                elif percent_diff <= 0.45:
                    self.last_env_index = num_envs - 1
            elif self.last_env_index == num_envs - 1:
                if percent_diff >= 0.6:
                    self.last_env_index = 1

        elif self.mode_env == 2:
            if length_percentages is None:
                raise Exception("length_percentage cannot be None in mode 2.")

            avg_length_percentage = np.mean(length_percentages)

            if self.last_env_index == 0:
                if avg_length_percentage >= 0.35:
                    self.last_env_index = 1
            elif self.last_env_index == 1:
                if avg_length_percentage <= 0.2:
                    self.last_env_index = 0
                elif avg_length_percentage >= 0.5:
                    self.last_env_index = num_envs - 1
            elif self.last_env_index == num_envs - 1:
                if avg_length_percentage <= 0.35:
                    self.last_env_index = 1

        elif self.mode_env == 3:
            percent_diffs = [
                abs(angles[0] - angles[1]) / angles[1] for angles in angles
            ]
            # take the mean of the lowest 3/4 of the percentages
            percent_diff = np.mean(
                sorted(percent_diffs)[: int(len(percent_diffs) * 3 / 4)]
            )

            # with open("angles.txt", "a") as f:
            #     f.write(str(percent_diff) + "\n")

            if self.last_env_index == 0:
                if percent_diff <= 0.04:
                    self.last_env_index = 1
            elif self.last_env_index == 1:
                if percent_diff >= 0.05:
                    self.last_env_index = 0
                elif percent_diff <= 0.02:
                    self.last_env_index = num_envs - 1
            elif self.last_env_index == num_envs - 1:
                if percent_diff >= 0.03:
                    self.last_env_index = 1

        elif self.mode_env == 4:
            vel_percent_diffs = [
                abs(velocity[0] - velocity[1]) / velocity[1]
                for velocity in velocities
            ]
            angle_percent_diffs = [
                abs(angle[0] - angle[1]) / angle[1] for angle in angles
            ]

            vel_percent_diff = np.mean(
                sorted(vel_percent_diffs)[
                    : int(len(vel_percent_diffs) * 3 / 4)
                ]
            )
            angle_percent_diff = np.mean(
                sorted(angle_percent_diffs)[
                    : int(len(angle_percent_diffs) * 3 / 4)
                ]
            )

            if self.last_env_index == 0:
                if vel_percent_diff <= 0.65 and angle_percent_diff <= 0.04:
                    self.last_env_index = 1
            elif self.last_env_index == 1:
                if vel_percent_diff >= 0.8 or angle_percent_diff >= 0.05:
                    self.last_env_index = 0
                elif vel_percent_diff <= 0.45 and angle_percent_diff <= 0.02:
                    self.last_env_index = num_envs - 1
            elif self.last_env_index == num_envs - 1:
                if vel_percent_diff >= 0.6 or angle_percent_diff >= 0.03:
                    self.last_env_index = 1

        elif self.mode_env == 5:
            if steps_per >= 0.2:
                self.last_env_index = 1
            elif steps_per >= 0.4:
                self.last_env_index = num_envs - 1

        if self.last_env_index != old_env_index:
            self.no_switch = 5

        # with open("angles.txt", "a") as f:
        #     f.write(str(angles) + "\n")

        # with open("velocities.txt", "a") as f:
        #     f.write(str(velocities) + "\n")

        return self.last_env_index, self.last_angle_range, self.last_vel_range
