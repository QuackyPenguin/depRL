import numpy as np

from deprl.vendor.tonic import logger
from deprl.vendor.tonic.replays import Buffer
from deprl.sampling_grid import SamplingGrid


class CurriculumBufferBWR(Buffer):
    """
    Assume all activity is appended at the end of the observation.
    True for Myosuite and scone so far.
    """

    def __init__(self, *args, **kwargs):
        # initialize the environment index, angle range, velocity range
        self.last_env_index = 1 # 0 - 4 year old, 1 - adult
        initial_angle_range = (0, 0)
        initial_vel_range = (0,0)
        self.resolution = (0.1, np.pi/8)
        success_threshold = 1250
        self.task = "standing"
        self.epoch_counter = 0

        # 0 - target task, 1 - velocity task, 2 - orientation task
        # initial task is for the 4 year old
        self.last_task = 1

        # initialize the grid adaptive curriculum
        self.sampling_grid = SamplingGrid(
            vel_range=initial_vel_range, 
            angle_range=initial_angle_range, 
            resolution=self.resolution, 
            success_threshold=success_threshold, 
        )

        # get the mode for switching the environment and the targets (angle, velocity, standing)
        self.mode_env = kwargs.pop("mode_env", 0)
        self.mode_target = kwargs.pop("mode_target", 0)

        if self.mode_env not in [0, 1, 2]:
            raise Exception(
                f"Mode {self.mode_env} of the environment is not implemented."
            )
        if self.mode_target not in [0, 1, 2, 3, 4, 5, 6, 7, 100, 110, 120, 130, 135, 140, 146]:
            raise Exception(
                f"Mode {self.mode_target} of the targets is not implemented."
            )
        
        # set settings for environment curriculum methods
        self.env_0_threshold = 0.4 # percentage of steps in the 4-year-old environment
        self.env_1_threshold = 2000 # threshold for the mean reward to switch to the adult environment

        # Initial values ----------------
        super().__init__(*args, **kwargs)

    def _curriculum_step(
        self,
        steps_per=0,
        reward_scale=1,
        collected_velocities=None,
        collected_angles=None,
        worker_rewards=None,
    ):
        """Perform a curriculum step. Update the environment index, angle range, and velocity range.

        Args:
            collected_velocities (list): The list of the tuples of the current and target velocities of the tasks; format: [(global_worker_id, episode_index, (normalized_actual_velocity, normalized_target_velocity))]_i.
            collected_angles (list): The list of the tuples of the current and target angles of the tasks; format: [(global_worker_id, episode_index, (actual_angle, target_angle))]_i.
            worker_rewards (dictionary): The list of the rewards of the tasks.

        Returns:
            int: The index of the environment to use.
            list: The new sampling grid.
            int: The task to be selected.
        """
        if worker_rewards is None:
            raise Exception(
                "rewards cannot be None to perform a curriculum step."
            )
        if collected_angles is None:
            raise Exception(
                "angles cannot be None to perform a curriculum step."
            )
        if collected_velocities is None:
            raise Exception(
                "velocities cannot be None to perform a curriculum step."
            )
        
        # get the data (target velocity and angle, actual velocity and angle, episode reward) from the workers for all episodes in the epoch
        episode_data = self._set_up_episode_data(collected_velocities, collected_angles, worker_rewards)

        # #### comment in if you want to plot the numbers of episodes at each node in the sampling grid after each epoch
        # #### tmp_counts: counts of episodes at each node in the sampling grid for the current epoch
        # #### total_counts: counts of episodes at each node in the sampling grid for all epochs so far
        # for global_worker_id, data in episode_data.items():
        #         for episode, tmp_data in data.items():
        #             target_vel = tmp_data['velocity'][0][1]
        #             target_angle = tmp_data['angle'][0][1]
        #             self.sampling_grid.add_episode_to_counts(target_vel, target_angle)
        #             self.sampling_grid.add_episode_to_counts(target_vel, target_angle,tmp=True)
        # self.sampling_grid.plot_counts(tmp=True,save_path=f"/home/nadinebadie/denis/valentin_results/test0/tmp_counts/{self.epoch_counter}.png")
        # self.sampling_grid.reset_tmp_counts()
        # self.sampling_grid.plot_counts(tmp=False,save_path=f"/home/nadinebadie/denis/valentin_results/test0/total_counts/{self.epoch_counter}.png")

        # #### comment in if you want to plot the sampling grid after each epoch
        # self.sampling_grid.plot(save_path=f"/home/nadinebadie/denis/valentin_results/test0/sampling_grid/{self.epoch_counter}.png")
        # self.epoch_counter += 1

        #### Curriculum for the environment
        if self.mode_env == 0:
            # change the environments at a fixed percentage of steps
            self.last_task = 2
            if steps_per >= self.env_0_threshold:
                self.last_env_index = 1
                self.last_task = 1

        elif self.mode_env == 1:
            # change the environment based on the reward function
            # if the mean reward is above a threshold, switch to the adult environment
            rewards = []
            for global_worker_id, data in episode_data.items():
                for episode, tmp_data in data.items():
                    rewards.append(tmp_data['reward'])
            mean_reward = np.mean(rewards)
            if mean_reward >= self.env_1_threshold:
                self.last_env_index = 1

        elif self.mode_env == 2:
            # no environment curriculum
            print(f"No environment curriculum, using environment {self.last_env_index} for all epochs.")

        
        #### Curriculum for the targets (target angle, target velocity)
        #####DENIS IMPLEMENTATION (DOES CURRENTLY NOT WORK ANYMORE, SINCE THE VELOCITIES AND ANGLES ARE STORED WITH WORKER AND EPISODE INFORMATION)
        target_0_threshold = [0.5,0.3] #[0.3,0.2] #[0.5, 0.15] #0.6
        # target_1_threshold[0] = env_1_threshold, so that the ranges are not changed until the environment is switched to the adult
        target_1_threshold =[0.08, 0.16, 0.24, 0.32, 0.4] #[0.08, 0.16, 0.4] # for 2.5e7 steps total until 1e7 in 4-year-old (B-W-R) then adult (R)
        # target_2_threshold = [0.5,0.3,750]
        target_2_threshold = [0.5,0.3,750]#, 1000] #1000: max reward
        # target_3_threshold is saved in the grid adaptive curriculum

        # print('reward_scale buffer', reward_scale)
        if self.mode_target == 0:
            # THIS MODE DOES CURRENTLY NOT WORK ANYMORE, SINCE THE VELOCITIES AND ANGLES ARE STORED WITH WORKER AND EPISODE INFORMATION
            # FURTHERMORE, SAMPLING IS NOT DONE FROM THE RANGES ANYMORE, INSTEAD SAMPLING IS DONE FROM THE SAMPLING GRID
            # increase the velocity range if the average difference between the current and target velocities is below a threshold
            print("velocity shape",np.shape(velocities))
            vel_percent_diffs = [
                abs(velocity[0] - velocity[1])/reward_scale #/ (velocity[1] + 0.0001)
                for velocity in velocities
            ]
            # Velocities and angles in all epoch steps (length is 2e5)
            # print('len(velocities)', len(velocities))
            
            # print('velocity[1]', [velocity[1] for velocity in velocities])
            # print('velocity[0]', [velocity[0] for velocity in velocities])
            # increase the angle range if the average difference between the current and target angles is below a threshold
            angle_percent_diffs = [
                abs(angle[0] - angle[1]) / np.pi for angle in angles
            ]
            # print('angle[0]',[angle[0] for angle in angles])
            # print('angle[1]',[angle[1] for angle in angles])

            print('len(angles)', len(angles))

            # only take the lowest 3/4 of the percentages, to avoid initial outliers
            vel_percent_diff = np.mean(vel_percent_diffs)
            # np.mean(
            #     vel_percent_diffs[: int(len(vel_percent_diffs) * 3 / 4)]
            # )

            # print('vel_percent_diff', vel_percent_diff)
            angle_percent_diff = np.mean(angle_percent_diffs)
            # np.mean(
            #     angle_percent_diffs[: int(len(angle_percent_diffs) * 3 / 4)]
            # )
            # print('angle_percent_diff', angle_percent_diff)

            if vel_percent_diff <= target_0_threshold[0]:
                vel_percent = min(1.25, self.last_vel_range[1] + 0.2) #0.05
                self.last_vel_range = (0, vel_percent) #(0.25, vel_percent)

                print('vel_percent change', vel_percent)

            if angle_percent_diff <= target_0_threshold[1]:
                angle_percent = min(
                    np.pi, self.last_angle_range[1] + np.pi / 18 #16 #32
                )
                self.last_angle_range = (-angle_percent, angle_percent)

                print('angle_percent change', angle_percent)

        elif self.mode_target == 1:
            # THIS MODE DOES CURRENTLY NOT WORK ANYMORE, SINCE THE VELOCITIES AND ANGLES ARE STORED WITH WORKER AND EPISODE INFORMATION
            # FURTHERMORE, SAMPLING IS NOT DONE FROM THE RANGES ANYMORE, INSTEAD SAMPLING IS DONE FROM THE SAMPLING GRID
            # increase the ranges based on the number of steps, but not simultaneously
            if steps_per <= target_1_threshold[0]:
                vel_percent = min(1, (steps_per - target_1_threshold[0]) / 0.2)
                self.last_vel_range = (
                    1 - 0.75 * vel_percent,
                    1 + 0.25 * vel_percent,
                )
                # while increasing the velocity range, the velocity task is selected
                self.last_task = 1

            if steps_per <= target_1_threshold[1] and steps_per > target_1_threshold[0]:
                angle_percent = min(
                    1, (steps_per - target_1_threshold[1]) / 0.2
                )
                self.last_angle_range = (
                    -angle_percent * np.pi,
                    angle_percent * np.pi,
                )
                # while increasing the angle range, the orientation task is selected
                self.last_task = 2

            if steps_per > target_1_threshold[1] and steps_per <= target_1_threshold[2]:
                self.last_task = 3

            if steps_per > target_1_threshold[2] and steps_per <= target_1_threshold[3]:
                self.last_task = 4
            
            if steps_per > target_1_threshold[3] and steps_per <= target_1_threshold[4]:
                self.last_task = 5
            if steps_per > target_1_threshold[4]:
                self.last_task = 0

        elif self.mode_target == 2:
            # increase the velocity and angle range if the mean reward is above a threshold and the mean difference between the current and target velocities and angles is below a threshold
            vel_percent_diffs = [
                abs(velocity[0] - velocity[1])/reward_scale
                for velocity in velocities
            ]

            angle_percent_diffs = [
                abs(angle[0] - angle[1]) / np.pi 
                for angle in angles
            ]

            vel_percent_diff = np.mean(vel_percent_diffs)

            angle_percent_diff = np.mean(angle_percent_diffs)

            mean_reward = np.mean(rewards)
            
            if mean_reward >= target_2_threshold[2]:
                if vel_percent_diff <= target_2_threshold[0]:
                    vel_percent = min(1.25, self.last_vel_range[1] + 0.1)
                    self.last_vel_range = (0, vel_percent)

                if angle_percent_diff <= target_2_threshold[1]:
                    angle_percent = min(np.pi, self.last_angle_range[1] + np.pi / 36)
                    self.last_angle_range = (-angle_percent, angle_percent)  
        
        elif self.mode_target == 100:
            # Dictionary to store episode data
            episode_data = self._set_up_episode_data(collected_velocities, collected_angles, worker_rewards)
            if self.task == "standing":
                mean_reward = np.mean([tmp_data['reward'] for worker_data in episode_data.values() for tmp_data in worker_data.values()])
                print("mean reward", mean_reward)
                if mean_reward >= 2100:#2000
                    self.task = "walking"
                    self.sampling_grid._extend_grid_velocity(skip=2)
                    print("switch to walking")
                    max_angle = 0
                    while max_angle < np.pi-1e-3:
                        self.sampling_grid._extend_grid_angle_top()
                        self.sampling_grid._extend_grid_angle_bottom()
                        angles = self.sampling_grid.grid[:, 1]
                        max_angle = np.max(angles)
                    self.sampling_grid._adapt_weights_03_04()
                    for i in range(len(self.sampling_grid.grid)):
                        if self.sampling_grid.grid[i][0] == 0:
                            self.sampling_grid.weights[i] = 0
            # elif self.task == "walking":
            #     mean_reward = np.mean([tmp_data['reward'] for worker_data in episode_data.values() for tmp_data in worker_data.values()])
            #     if mean_reward >= 1500:
            #         self.last_env_index = 1

        elif self.mode_target == 110:
            episode_data = self._set_up_episode_data(collected_velocities, collected_angles, worker_rewards)

            #track episode vels and angles
            for global_worker_id, data in episode_data.items():
                for episode, tmp_data in data.items():
                    target_vel = tmp_data['velocity'][0][1]
                    target_angle = tmp_data['angle'][0][1]
                    self.sampling_grid.add_episode_to_counts(target_vel, target_angle)
                    self.sampling_grid.add_episode_to_counts(target_vel, target_angle,tmp=True)
            self.sampling_grid.plot_counts(tmp=True,savepath=f"/home/nadinebadie/denis/valentin_results/test_19_0/tmp_counts/{self.epoch_counter}.png")
            self.sampling_grid.reset_tmp_counts()
            self.sampling_grid.plot_counts(tmp=False,savepath=f"/home/nadinebadie/denis/valentin_results/test_19_0/total_counts/{self.epoch_counter}.png")
            self.epoch_counter += 1

            if self.task == "standing":
                mean_reward = np.mean([tmp_data['reward'] for worker_data in episode_data.values() for tmp_data in worker_data.values()])
                print("mean reward", mean_reward)
                if mean_reward >= 2100:
                    self.task = "walking"
                    self.sampling_grid._extend_grid_velocity(skip=2)
                    self.sampling_grid._adapt_weights_03_04()
                    print("switch to walking")
                    # max_angle = 0
                    # while max_angle < np.pi-1e-3:
                    #     self.sampling_grid._extend_grid_angle_top()
                    #     self.sampling_grid._extend_grid_angle_bottom()
                    #     angles = self.sampling_grid.grid[:, 1]
                    #     max_angle = np.max(angles)
                    # self.sampling_grid._adapt_weights_03_04()
                    # for i in range(len(self.sampling_grid.grid)):
                    #     if self.sampling_grid.grid[i][0] == 0:
                    #         self.sampling_grid.weights[i] = 0
            elif self.task == "walking":
                for global_worker_id, data in episode_data.items():
                    for episode, tmp_data in data.items():
                        target_vel = tmp_data['velocity'][0][1]
                        target_angle = tmp_data['angle'][0][1]
                        reward = tmp_data['reward']
                        self.sampling_grid.update_08_04_fixed_weights_angle(target_vel,target_angle,reward)
                self.sampling_grid._adapt_weights_03_04()
                for i in range(len(self.sampling_grid.grid)):
                    if self.sampling_grid.grid[i][0] == 0:
                        self.sampling_grid.weights[i] = 0
        
        elif self.mode_target == 120:
            episode_data = self._set_up_episode_data(collected_velocities, collected_angles, worker_rewards)

            #  #track episode vels and angles
            # for global_worker_id, data in episode_data.items():
            #     for episode, tmp_data in data.items():
            #         target_vel = tmp_data['velocity'][0][1]
            #         target_angle = tmp_data['angle'][0][1]
            #         self.sampling_grid.add_episode_to_counts(target_vel, target_angle)
            #         self.sampling_grid.add_episode_to_counts(target_vel, target_angle,tmp=True)
            # self.sampling_grid.plot_counts(tmp=True,savepath=f"/home/nadinebadie/denis/valentin_results/results_21/tmp_counts/{self.epoch_counter}.png")
            # self.sampling_grid.reset_tmp_counts()
            # self.sampling_grid.plot_counts(tmp=False,savepath=f"/home/nadinebadie/denis/valentin_results/results_21/total_counts/{self.epoch_counter}.png")
            # self.epoch_counter += 1

            if self.task == "standing":
                mean_reward = np.mean([tmp_data['reward'] for worker_data in episode_data.values() for tmp_data in worker_data.values()])
                print("mean reward", mean_reward)
                if mean_reward >= 2100:
                    self.task = "walking"
                    self.sampling_grid._extend_grid_velocity(skip=2)
                    self.sampling_grid._adapt_weights_03_04()
                    self.sampling_grid.weights[0] = 0
                    print("switch to walking")
                    max_vel = 0
                    while max_vel < 1.2-1e-3:
                        self.sampling_grid._extend_grid_velocity()
                        vels = self.sampling_grid.grid[:, 0]
                        max_vel = np.max(vels)
                    self.sampling_grid._adapt_weights_03_04()
                    for i in range(len(self.sampling_grid.grid)):
                        if self.sampling_grid.grid[i][0] == 0:
                            self.sampling_grid.weights[i] = 0        
            # elif self.task == "walking":
            #     for global_worker_id, data in episode_data.items():
            #         for episode, tmp_data in data.items():
            #             target_vel = tmp_data['velocity'][0][1]
            #             target_angle = tmp_data['angle'][0][1]
            #             reward = tmp_data['reward']
            #             self.sampling_grid.update_03_04_fixed_weights(steps_per,target_vel,target_angle,reward)
            #             self.sampling_grid.weights[0] = 0
        
        elif self.mode_target == 130:
            episode_data = self._set_up_episode_data(collected_velocities, collected_angles, worker_rewards)
            
            # #track episode vels and angles
            # for global_worker_id, data in episode_data.items():
            #     for episode, tmp_data in data.items():
            #         target_vel = tmp_data['velocity'][0][1]
            #         target_angle = tmp_data['angle'][0][1]
            #         self.sampling_grid.add_episode_to_counts(target_vel, target_angle)
            #         self.sampling_grid.add_episode_to_counts(target_vel, target_angle,tmp=True)
            # self.sampling_grid.plot_counts(tmp=True,savepath=f"/home/nadinebadie/denis/valentin_results/results_31-0-try2/tmp_counts/{self.epoch_counter}.png")
            # self.sampling_grid.reset_tmp_counts()
            # self.sampling_grid.plot_counts(tmp=False,savepath=f"/home/nadinebadie/denis/valentin_results/results_31-0-try2/total_counts/{self.epoch_counter}.png")
            # self.epoch_counter += 1

            if self.task == "standing":
                mean_reward = np.mean([tmp_data['reward'] for worker_data in episode_data.values() for tmp_data in worker_data.values()])
                print("mean reward", mean_reward)
                if mean_reward >= 2100:
                    self.task = "walking"
                    self.sampling_grid._extend_grid_velocity(skip=2)
                    self.sampling_grid._adapt_weights_03_04()
                    self.sampling_grid.weights[0] = 0
                    print("switch to walking")
                    for i in range(len(self.sampling_grid.grid)):
                        if self.sampling_grid.grid[i][0] == 0:
                            self.sampling_grid.weights[i] = 0      
            elif self.task == "walking":
                for global_worker_id, data in episode_data.items():
                    for episode, tmp_data in data.items():
                        target_vel = tmp_data['velocity'][0][1]
                        target_angle = tmp_data['angle'][0][1]
                        reward = tmp_data['reward']
                        self.sampling_grid.update_08_04_fixed_weights_angle(target_vel,target_angle,reward)
                        for i in range(len(self.sampling_grid.grid)):
                            if self.sampling_grid.grid[i][0] == 0:
                                self.sampling_grid.weights[i] = 0
                if steps_per > 2/3:
                        self.task = "learn_different_speeds"
                        print("switch to learn_different_speeds")
            elif self.task == "learn_different_speeds":
                transformed_steps_per = steps_per - 2/3
                transformed_steps_per = transformed_steps_per / (1 - 2/3)
                self.sampling_grid.update_fixed(transformed_steps_per)
                for i in range(len(self.sampling_grid.grid)):
                    if self.sampling_grid.grid[i][0] == 0:
                        self.sampling_grid.weights[i] = 0


        elif self.mode_target == 135:
            if steps_per >= 0.1:
                self.last_env_index = 1
            # Dictionary to store episode data
            episode_data = self._set_up_episode_data(collected_velocities, collected_angles, worker_rewards)
           
            if self.task == "standing":
                mean_reward = np.mean([tmp_data['reward'] for worker_data in episode_data.values() for tmp_data in worker_data.values()])
                print("mean reward", mean_reward)
                if mean_reward >= 2000:
                    self.task = "walking"
                    self.sampling_grid._extend_grid_velocity(skip=2)
                    self.sampling_grid._adapt_weights()
                    self.sampling_grid.weights[0] = 0
                    print("switch to walking")
            elif self.task == "walking":
                # Maximum velocity and angle
                max_velocity = np.max(self.sampling_grid.grid[:, 0])
                max_angle = np.max(self.sampling_grid.grid[:, 1])
                min_angle = np.min(self.sampling_grid.grid[:, 1])
                print("max velocity", max_velocity)
                print("max angle", max_angle)
                print ("min angle", min_angle)

                # Tolerance for "closeness"
                tolerance = 1e-2
       
                #Filter episodes (version of meeting on 12.02.2024)
                #put all episodes to the filtered dict, are on the edges and close to the corners
                filtered_episodes = {}
                tolerance_vel = 1.5 * self.resolution[0]
                tolerance_angle = 1.5 * self.resolution[1]
                for global_worker_id, data in episode_data.items():
                    for episode, tmp_data in data.items():
                        target_vel = tmp_data['velocity'][0][1]
                        target_angle = tmp_data['angle'][0][1]
                        if ((abs(target_angle - max_angle) < tolerance_angle or abs(target_angle - min_angle) < tolerance_angle) and abs(target_vel - max_velocity) < tolerance_vel):
                            if abs(target_vel - max_velocity) < tolerance or abs(target_angle - max_angle) < tolerance or abs(target_angle - min_angle) < tolerance:
                                if global_worker_id not in filtered_episodes:
                                    filtered_episodes[global_worker_id] = {}
                                # print("Target velocity: ", target_vel, "Target angle: ", target_angle, "worker: ", global_worker_id, "episode: ", episode, "max velocity: ", max_velocity, "max angle: ", max_angle)
                                filtered_episodes[global_worker_id][episode] = tmp_data
                                reward = tmp_data['reward']
                                self.sampling_grid.update(
                                    steps_per, target_vel, target_angle, reward
                                    )



        elif self.mode_target == 140:
            episode_data = self._set_up_episode_data(collected_velocities, collected_angles, worker_rewards)
            
            # #track episode vels and angles
            # for global_worker_id, data in episode_data.items():
            #     for episode, tmp_data in data.items():
            #         target_vel = tmp_data['velocity'][0][1]
            #         target_angle = tmp_data['angle'][0][1]
            #         self.sampling_grid.add_episode_to_counts(target_vel, target_angle)
            #         self.sampling_grid.add_episode_to_counts(target_vel, target_angle,tmp=True)
            # self.sampling_grid.plot_counts(tmp=True,savepath=f"/home/nadinebadie/denis/valentin_results/results_41-0/tmp_counts/{self.epoch_counter}.png")
            # self.sampling_grid.reset_tmp_counts()
            # self.sampling_grid.plot_counts(tmp=False,savepath=f"/home/nadinebadie/denis/valentin_results/results_41-0/total_counts/{self.epoch_counter}.png")
            # self.epoch_counter += 1

            if self.task == "standing":
                mean_reward = np.mean([tmp_data['reward'] for worker_data in episode_data.values() for tmp_data in worker_data.values()])
                print("mean reward", mean_reward)
                if mean_reward >= 2000:
                    self.task = "walking"
                    self.sampling_grid._extend_grid_velocity(skip=2)
                    self.sampling_grid._adapt_weights_03_04()
                    self.sampling_grid.weights[0] = 0
                    print("switch to walking")
                    for i in range(len(self.sampling_grid.grid)):
                        if self.sampling_grid.grid[i][0] == 0:
                            self.sampling_grid.weights[i] = 0      
            elif self.task == "walking":
                for global_worker_id, data in episode_data.items():
                    for episode, tmp_data in data.items():
                        target_vel = tmp_data['velocity'][0][1]
                        target_angle = tmp_data['angle'][0][1]
                        reward = tmp_data['reward']
                        self.sampling_grid.update_08_04_fixed_weights_angle(target_vel,target_angle,reward)
                        for i in range(len(self.sampling_grid.grid)):
                            if self.sampling_grid.grid[i][0] == 0:
                                self.sampling_grid.weights[i] = 0
                if steps_per > 1/3:
                        self.task = "learn_different_speeds"
                        print("switch to learn_different_speeds")
            elif self.task == "learn_different_speeds":
                transformed_steps_per = steps_per - 1/3
                transformed_steps_per = transformed_steps_per / (1 - 1/3)
                self.sampling_grid.update_fixed(transformed_steps_per)
                for i in range(len(self.sampling_grid.grid)):
                    if self.sampling_grid.grid[i][0] == 0:
                        self.sampling_grid.weights[i] = 0
                mean_reward = np.mean([tmp_data['reward'] for worker_data in episode_data.values() for tmp_data in worker_data.values()])
                print("mean reward", mean_reward)
                max_vel = np.max(self.sampling_grid.grid[:, 0])
                if max_vel >= 0.5 - 1e-3 and mean_reward >= 1600:
                    self.last_env_index= 1

        elif self.mode_target == 146:
            print("do nothing")

        return (
            self.last_env_index,
            self.last_task,
            self.sampling_grid
        )

    def _set_up_episode_data(self, collected_velocities, collected_angles, worker_episode_rewards):
        episode_data = {}
        # Loop through the entries and populate the nested dictionary with velocities
        for global_worker_id, episode, velocity in collected_velocities:
            if global_worker_id not in episode_data:
                episode_data[global_worker_id] = {}  # Create dictionary for the global_worker_id

            # Ensure the inner key (episode) exists
            if episode not in episode_data[global_worker_id]:
                episode_data[global_worker_id][episode] = {
                    'velocity': [],  
                    'angle': [],
                    'reward': None
                }

            episode_data[global_worker_id][episode]['velocity'].append(velocity)
        # Loop through the entries and populate the nested dictionary with angles
        for global_worker_id, episode, angle in collected_angles:
            episode_data[global_worker_id][episode]['angle'].append(angle)

        # Add rewards for each episode
        for global_worker_id, rewards in worker_episode_rewards.items():
            for episode_id, reward in enumerate(rewards):
                if episode_id in episode_data[global_worker_id]:
                    episode_data[global_worker_id][episode_id]['reward'] = reward
                elif episode_id not in episode_data[global_worker_id] and episode_id == len(rewards) - 1 and abs(reward) < 1e-3:
                    pass
                else:
                    raise ValueError(f"Episode {episode_id} not found for worker {global_worker_id}")

        # remove episodes that are still running
        for global_worker_id in range(len(episode_data.keys())):
            if worker_episode_rewards[global_worker_id][-1] != 0:
                del episode_data[global_worker_id][max(episode_data[global_worker_id].keys())]

        # Convert velocities and angles to numpy arrays for consistency
        for global_worker_id, data in episode_data.items():
            for episode, tmp_data in data.items():
                tmp_data['velocity'] = np.array(tmp_data['velocity'])
                tmp_data['angle'] = np.array(tmp_data['angle'])            

        return episode_data



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