import numpy as np
import matplotlib.pyplot as plt

class SamplingGrid:
    def __init__(self, vel_range=(-1.0, 1.0), angle_range=(-1, 1), resolution=(0.1, np.pi/8), success_threshold=1000):
        self.resolution_vel = resolution[0]
        self.resolution_angle = resolution[1]
        self.success_threshold = success_threshold
        self._grid = self._create_grid(vel_range, angle_range, self.resolution_vel, self.resolution_angle)

        # counter for reward-based updates
        self.success_counter_vel = 0
        self.success_counter_angle_top = 0
        self.success_counter_angle_bottom = 0

        # counter for fixed updates
        self.num_of_updates_vel = None
        self.num_of_updates_angle = None
        self.curr_num_of_updates_vel = 0
        self.curr_num_of_updates_angle = 0

        # only used for plotting: initialize counts for plotting the number of episodes in each cell
        self.counts = {}
        self.tmp_counts = {}

    @property
    def weights(self):
        return self._grid[:, 2]
    
    @property
    def grid(self):
        return self._grid[:, :2]
    
    def set_success_threshold(self, success_threshold):
        self.success_threshold = success_threshold

####################################################################################################################################
####################################################################################################################################
# functions for creating and extending the grid
####################################################################################################################################
####################################################################################################################################
    def _create_grid(self, vel_range, angle_range, resolution_vel, resolution_angle):
        """
        Creates a grid of points in the velocity-angle space. Each point is initialized with a weight of 1.

        Args:
            vel_range (tuple): The range of velocity values (min, max).
            angle_range (tuple): The range of angle values (min, max).
            resolution_vel (float): The resolution of the grid in velocity direction.
            resolution_angle (float): The resolution of the grid in angle direction.

        Returns:
            numpy.ndarray: The grid in Row-Major Order as a 2D array (n, 3), where n is the number of points. 
                           First two columns are velocity and angle, and the third column is the weight.
        """
        vel_values = np.arange(vel_range[0], vel_range[1] + resolution_vel/10, resolution_vel)
        angle_values = np.arange(angle_range[0], angle_range[1] + resolution_angle/10, resolution_angle)
        grid = np.array([[vel, angle, 1] for vel in vel_values for angle in angle_values])

        for vel in vel_values:
            for angle in angle_values:
                self.counts[(vel, angle)] = 0
                self.tmp_counts[(vel, angle)] = 0

        return grid
    
    def extend_grid_velocity(self, skip=0):
        """
        Extends the grid along the velocity axis (x-axis) in positive direction.
        The new points are created by incrementing the maximum velocity in the grid by the resolution of the velocity.
        The weights of the new points are initialized to 0.

        Args:
            skip (int): The number of resolution steps to skip when extending the grid. Default is 0.
        
        Returns:
            numpy.ndarray: The extended grid in Row-Major Order as a 2D array (n, 3).
        """
        # Extracting the maximum velocity from the existing grid
        max_velocity = np.max(self.grid[:, 0])
        angle_values = np.unique(self.grid[:, 1])

        # Creating the new points for the extension
        max_velocity += self.resolution_vel * (skip + 1)
        new_points = np.array([[max_velocity, a, 0] for a in angle_values])

        # Extending the grid with the new points
        self._grid = np.row_stack((self._grid[:, :], new_points))

        # initialize the counts for the new points (for plotting the number of episodes in each cell) 
        for angle in angle_values:
            self.counts[(max_velocity, angle)] = 0
            self.tmp_counts[(max_velocity, angle)] = 0

        print("Extended grid along positive velocity axis.")
        if skip > 0:
            print(f"Skipped {skip} steps from max velocity {max_velocity-self.resolution_vel*(skip+1)} to {max_velocity}.")

    def extend_grid_angle_top(self):
        """
        Extends the grid along the angle axis (y-axis) in positive direction.
        The new points are created by incrementing the maximum angle in the grid by the resolution of the angle.
        The weights of the new points are initialized to 0.
        
        Args:
            None

        Returns:
            numpy.ndarray: The extended grid in Row-Major Order as a 2D array (n, 3).
        """
        # Extracting the maximum angle from the existing grid
        max_angle = np.max(self.grid[:, 1])
        velocity_values = np.unique(self.grid[:, 0])
        angle_values = np.unique(self.grid[:, 1])

        # Creating the new points for the extension
        max_angle += self.resolution_angle
        new_points = np.array([[v, max_angle, 0] for v in velocity_values])

        new_point_indices = np.arange(len(angle_values), len(self.grid) + len(angle_values), len(angle_values))

        # Extending the grid with the new points
        self._grid = np.insert(self._grid, new_point_indices, new_points, axis=0)  

        # initialize the counts for the new points (for plotting the number of episodes in each cell)
        for velocity in velocity_values:
            self.counts[(velocity, max_angle)] = 0
            self.tmp_counts[(velocity, max_angle)] = 0

        print("Extended grid along positive angle axis")

    def extend_grid_angle_bottom(self):
        """
        Extends the grid along the angle axis (y-axis) in negative direction.
        The new points are created by decrementing the minimum angle in the grid by the resolution of the angle.
        The weights of the new points are initialized to 0.

        Args:
            None

        Returns:
            numpy.ndarray: The extended grid in Row-Major Order as a 2D array (n, 3).
        """
        # Extracting the maximum and minimum angles from the existing grid
        min_angle = np.min(self.grid[:, 1])
        velocity_values = np.unique(self.grid[:, 0])
        angle_values = np.unique(self.grid[:, 1])

        # Creating the new points for the extension
        min_angle -= self.resolution_angle
        new_points = np.array([[v, min_angle, 0] for v in velocity_values])

        new_point_indices = np.arange(0, len(self.grid), len(angle_values))

        # Extending the grid with the new points
        self._grid = np.insert(self._grid, new_point_indices, new_points, axis=0)

        # initialize the counts for the new points (for plotting the number of episodes in each cell)
        for velocity in velocity_values:
            self.counts[(velocity, min_angle)] = 0
            self.tmp_counts[(velocity, min_angle)] = 0

        print("Extended grid along negative angle axis")

#####################################################################################################################################
#####################################################################################################################################
# functions for getting and checking nodes and their adjacents
#####################################################################################################################################
#####################################################################################################################################    
    def _get_node(self, velocity, angle):
        """
        Finds the closest grid point to the given velocity and angle.

        Args:
            velocity (float): The velocity value.
            angle (float): The angle value.
        Returns:
            tuple: The closest grid point (velocity, angle) and its index in the grid.
        """
        squared_distances = (self.grid[:, 0] - velocity) ** 2 + (self.grid[:, 1] - angle) ** 2
        index = np.argmin(squared_distances)
        return self.grid[index], index
    
    def _is_border_vel(self, index):
        """
        Checks if the given index corresponds to the maximum velocity in the grid.
        Args:
            index (int): The index of the grid point to check.
        Returns:
            bool: True if the index corresponds to the maximum velocity, False otherwise.
        """
        max_vel = np.max(self.grid[:, 0])
        if self.grid[index, 0] == max_vel:
            return True
        return False
    
    def _is_border_angle_top(self, index):
        """
        Checks if the given index corresponds to the maximum angle in the grid.
        Args:
            index (int): The index of the grid point to check.
        Returns:
            bool: True if the index corresponds to the maximum angle, False otherwise.
        """
        max_angle = np.max(self.grid[:, 1])
        if self.grid[index, 1] == max_angle:
            return True
        return False

    def _is_border_angle_bottom(self, index):
        """
        Checks if the given index corresponds to the minimum angle in the grid.
        Args:
            index (int): The index of the grid point to check.
        Returns:
            bool: True if the index corresponds to the minimum angle, False otherwise.
        """
        min_angle = np.min(self.grid[:, 1])
        if self.grid[index, 1] == min_angle:
            return True
        return False
    
    def _get_adjacents(self, index):
        """
        Returns a boolean array indicating which grid points are adjacent to the point at the given index.
        Adjacent points are defined as those within 1.5 times the resolution in both velocity and angle dimensions.

        Args:
            index (int): The index of the grid point to check for adjacents.
        Returns:
            numpy.ndarray: A boolean array indicating which grid points are adjacent to the point at the given index.
        """
        resolution = np.array([self.resolution_vel, self.resolution_angle])
        adjacent_inds = np.logical_and(
            self.grid[:, :] >= self.grid[index, :] - 1.5*resolution,
            self.grid[:, :] <= self.grid[index, :] + 1.5*resolution
        ).all(axis=1)

        return adjacent_inds
    
######################################################################################################################################
######################################################################################################################################
# functions for updating the grid with the specified curriculum methods
######################################################################################################################################
######################################################################################################################################
    def update(self, update_method = None, weight_adaptation_method = "uniform", training_progress = None, target_velocity = None, target_angle = None, reward = None):
        """
        Updates the grid based on the specified update method.
        Args:
            update_method (str): The method to use for updating the grid. Currently possible options:
            - 'fixed': Extends the grid at fixed intervals based on training progress.
            - 'reward_based_angle': Extends the grid in angle direction based on the mean reward achieved in the last epoch. Keeps the velocity fixed.
            - 'reward_based_vel': Extends the grid in velocity direction based on the mean reward achieved in the last epoch. Keeps the angle fixed.
            - 'reward_based': Extends the grid in both velocity and angle directions based on the reward achieved in the last epoch.
            weight_adaptation_method (str): The method to use for adapting weights after the grid update. Default is 'uniform'.
            training_progress (float): The current training progress as a fraction between 0 and 1. Only necessary for the 'fixed' method.
            target_velocity (float): The target velocity of the last episode. 
            target_angle (float): The target angle of the last episode. 
            reward (float): The reward achieved in the last episode. 
        """
        if update_method == 'fixed':
            if training_progress == None: 
                raise Exception("training progress must be given to perform fixed grid updates")
            self._update_fixed(training_progress, weight_adaptation_method = weight_adaptation_method)
        elif update_method == 'reward_based_angle':
            if target_velocity == None or target_angle == None or reward == None:
                    raise Exception("target velocity, target angle and reward must not be None to perform reward-based grid updates") 
            self._update_reward_based(target_velocity, target_angle, reward, restrict_to = 'angle', weight_adaptation_method= weight_adaptation_method)
        elif update_method == 'reward_based_vel':
            if target_velocity == None or target_angle == None or reward == None:
                    raise Exception("target velocity, target angle and reward must not be None to perform reward-based grid updates") 
            self._update_reward_based(target_velocity, target_angle, reward, restrict_to= 'vel', weight_adaptation_method= weight_adaptation_method)
        elif update_method == 'reward_based':
            if target_velocity == None or target_angle == None or reward == None:
                    raise Exception("target velocity, target angle and reward must not be None to perform reward-based grid updates") 
            self._update_reward_based(training_progress, target_velocity, target_angle, reward, weight_adaptation_method= weight_adaptation_method)
        else:
            raise Exception(f"Update method {update_method} not implemented yet")
        
    def _update_fixed(self,training_progress, weight_adaptation_method, restrict_to = None, end_vel = 1.2, end_angle = np.pi):
        """
        Updates the grid based on the training progress.
        It extends the grid at fixed stages of training progress such that at the end of training, the grid covers the predefined ranges.
        Args:
            training_progress (float): The current training progress, ranging from 0 to 1.
            weight_adaptation_method (str): The method to use for adapting weights after the grid update.
            restrict_to (str, optional): If specified, restrict the grid update to the given dimension ('angle' or 'vel').
            end_vel (float): The target maximum velocity to reach by the end of training. Default is 1.2.
            end_angle (float): The target maximum angle to reach by the end of training. Default is np.pi.
        """
        velocities = self.grid[:, 0]
        max_vel = np.max(velocities)
        angles = self.grid[:, 1]
        max_angle = np.max(angles)
        min_angle = np.min(angles)
        if np.abs(max_angle - min_angle) > 1e-3:
            raise Exception("Grid is not symmetric in angle direction. Fixed updates only work for symmetric grids.")
        if max_vel > 1e-3 and max_vel < 0.3 - 1e-3:
            raise Exception("Currently fixed updates only work for max_vel = 0 or max_vel >= 0.3")
        # initialize the number of updates at the first call; number is calculated s.t. on init value (max_...) and end value we have some training time
        if self.num_of_updates_vel is None and self.num_of_updates_angle is None:
            self.end_vel = end_vel
            self.end_angle = end_angle
            # if the init velocity (max_vel) is 0, we need fewer update steps, since we skip 0.1 and 0.2 as target velocities
            self.num_of_updates_vel = int((end_vel-max_vel) / self.resolution_vel + 1) if max_vel > 0.3 - 1e-3 else int((end_vel) / self.resolution_vel - 1)
            self.num_of_updates_angle = int((end_angle - np.abs(min_angle)) / self.resolution_angle + 1)
        if self.end_vel != end_vel or self.end_angle != end_angle:
            raise Exception(f"end_vel and end_angle must not be changed after the first call of fixed updates. end_vel: {self.end_vel}, end_angle: {self.end_angle}")
        # Check if the grid needs to be extended in angle direction based on the training progress
        if restrict_to != 'vel':
            if training_progress >= (self.curr_num_of_updates_angle+1)/self.num_of_updates_angle:
                self.curr_num_of_updates_angle += 1
                if max_angle < np.pi:
                    self.extend_grid_angle_top()
                if min_angle > -np.pi:
                    self.extend_grid_angle_bottom()
        # Check if the grid needs to be extended in velocity direction based on the training progress
        if restrict_to != 'angle':
            if training_progress >= (self.curr_num_of_updates_vel+1)/self.num_of_updates_vel:
                self.curr_num_of_updates_vel += 1
                if max_vel < 1.25:
                    # we skip the target velocities 0.1 and 0.2, because they are especially difficult to learn
                    if max_vel - 0.25 < 1e-3:
                        self.extend_grid_velocity(skip= 2 - max_vel * 10)
                    else:
                        self.extend_grid_velocity()
        self.adapt_weights(weight_adaptation_method)

    def _update_reward_based(self, training_progress, velocity, angle, reward, weight_adaptation_method, restrict_to = None):
        """
        Updates the grid based on the reward achieved in the last episode.
        It extends the grid in velocity and/or angle directions if the reward exceeds the success threshold and the current point is at the border of the grid.
        Args:
            training_progress (float): The current training progress, ranging from 0 to 1.
            velocity (float): The target velocity of the episode.
            angle (float): The target angle of the episode.
            reward (float): The reward achieved in the last episode.
            weight_adaptation_method (str): The method to use for adapting weights after the grid update.
            restrict_to (str, optional): If specified, restrict the grid update to the given dimension ('angle' or 'vel').
        """
        if reward >= self.success_threshold:
            _, node_idx = self._get_node(velocity, angle)
            if restrict_to != 'angle':
                if self._is_border_vel(node_idx) and self.grid[node_idx, 0] < 1.25:
                    if self.success_counter_vel > 50:
                        self.extend_grid_velocity()
                        self.success_counter_vel = 0
                        print("Extended grid along velocity axis")
                    else:
                        self.success_counter_vel += 1
            if restrict_to != 'vel':
                if self._is_border_angle_top(node_idx) and self.grid[node_idx, 1] < np.pi:
                    if self.success_counter_angle_top > 50:
                        self.extend_grid_angle_top()
                        self.success_counter_angle_top = 0
                        print("Extended grid along angle top angle axis")
                    else:
                        self.success_counter_angle_top += 1
                if self._is_border_angle_bottom(node_idx) and self.grid[node_idx, 1] > -np.pi:
                    if self.success_counter_angle_bottom > 50:
                        self.extend_grid_angle_bottom()
                        self.success_counter_angle_bottom = 0
                        print("Extended grid along angle bottom axis")
                    else:
                        self.success_counter_angle_bottom += 1
        self.adapt_weights(weight_adaptation_method)

####################################################################################################################################
####################################################################################################################################
# functions for adapting the weights of the grid
####################################################################################################################################
####################################################################################################################################       
    def adapt_weights(self, weight_adaptation_method, index=None):
        """
        Adapts the weights of the grid based on the specified weight adaptation method.
        Args:
            index (int): The index of the grid point to adapt weights for. 
            weight_adaptation_method (str): The method to use for adapting weights. Currently possible options:
            - 'uniform': Sets all weights to 1.
            - 'more weights on corners': Computes new weights for whole grid (higher weights for points near corners and high-velocity points)
            - 'neighbors': Increases weights for the specified point and its adjacent points.
        """
        if weight_adaptation_method == "uniform":
            self._adapt_weights_uniform()
        elif weight_adaptation_method == "more weights on corners":
            self._adapt_weights_focus_on_corners()
        elif weight_adaptation_method == "neighbors":
            if index is None:
                raise Exception("Index must be provided for 'neighbors' weight adaptation method")
            self._adapt_weights_neighbors(index)
        else:
            raise Exception(f"Weight adaptation method {weight_adaptation_method} not implemented yet")

    def _adapt_weights_uniform(self):
        """
        Sets all weights in the grid to 1, making the sampling uniform across all grid points.
        """
        self._grid[:,2] = 1

    def _adapt_weights_focus_on_corners(self):
        """
        Adapts the weights of the grid points based on their proximity to the corners of the grid.
        The corners are defined as the top-right and bottom-right corners of the grid.
        The weights are a blend of the distance to the corners and the velocity importance.
        The closer a point is to the corners, the higher its weight, and the higher the velocity, the higher its weight.
        The weights are normalized to be between 0 and 1.
        """
        corners = [
            (np.max(self.grid[:, 0]), np.max(self.grid[:, 1])),  # Top-right corner
            (np.max(self.grid[:, 0]), np.min(self.grid[:, 1]))   # Bottom-right corner
        ]

        # Compute distance of each grid point to the nearest corner
        distances = np.array([
            min(np.linalg.norm(self.grid[i] - np.array(corner)) for corner in corners)
            for i in range(len(self.grid))
        ])

        # Normalize distances (0 = closest to corner, 1 = farthest from any corner)
        max_distance = np.max(distances)
        distances = distances / (max_distance + 1e-6)  # Avoid division by zero

        # Extract velocity values from the grid
        velocities = self.grid[:, 0]
        max_vel = np.max(velocities)

        # Scale velocity importance: Normalize velocity values between 0 and 1
        velocity_weights = velocities / (max_vel + 1e-6)  # Higher velocity → higher weight

        # Blend corner proximity and velocity importance with controlled scaling
        alpha = 0.6  # Adjusts the influence of corner proximity (higher = more corner focus)
        beta = 1 - alpha  # Ensures sum remains <= 1

        self._grid[:, 2] = alpha * (1 - distances) + beta * velocity_weights

    def _adapt_weights_neighbors(self, index):
        """
        Sets the weight of the node at the given index to 1 and sets the weights of 
        its adjacent nodes (in the 8-connected grid domain) to 1 as well.
        Args:
            index (int): The index of the grid point to update.
        """
        self._grid[index, 2] = 1
        adjacents = self._get_adjacents(index)
        adjacent_inds = np.array(adjacents.nonzero()[0])
        self._grid[adjacent_inds, 2] = 1

####################################################################################################################################
####################################################################################################################################
# functions for sampling from the grid
####################################################################################################################################
####################################################################################################################################
    def sample(self):
        """
        Samples a point uniformly from a cell around a randomly selected node in the grid.

        Returns:
            tuple: A tuple containing:
                - The sampled point from the selected cell.
                - The index of the selected node (the center of the cell).
        """
        center, index = self._sample_node()
        # return self._sample_uniform_from_cell(center), index
        return center, index
    
    def _sample_node(self):
        """
        Samples a node from the grid based on the weights of the nodes.
        Args:
            None
        Returns:
            tuple: The sampled node (velocity, angle) and its index in the grid.
        """
        index = np.random.choice(len(self.grid), 1, p=self.weights / self.weights.sum())
        return self.grid[index][0], index[0]
    
    def _sample_uniform_from_cell(self, center):
        """
        Samples a point uniformly from the cell centered at the given center.
        Cells are defined by the resolution in velocity and angle dimensions.
        Args:
            center (numpy.ndarray): The center of the cell.
        Returns:
            numpy.ndarray: A point sampled uniformly from the cell.
        """
        cell_sizes = np.array([self.resolution_vel, self.resolution_angle])
        low, high = center - cell_sizes / 2, center + cell_sizes / 2
        return np.random.uniform(np.array([np.max([0.,low[0]]),low[1]]), high)


#####################################################################################################################################
#####################################################################################################################################
# functions for plotting the grid and tracking + plotting the number of episodes in each cell
#####################################################################################################################################
#####################################################################################################################################
    def plot(self, title = "Grid with weights", label = "Grid points", save_path=None):
        """
        Plots the grid with weights as a scatter plot.
        The points are colored based on their weights, and a rectangle is drawn around the maximum grid area.
        Args:
            title (str): The title of the plot. Default is "Grid with weights".
            label (str): The label for the scatter plot. Default is "Grid points".
            save_path (str): The path to save the plot. If None, the plot is not saved.
        Returns:
            None
        """
        fig = plt.figure()
        scatter = plt.scatter(self.grid[:,0], self.grid[:,1], s=self.weights*100, c=self.weights, cmap='viridis', alpha=1, edgecolors='w', label = label)
        plt.gca().add_patch(plt.Rectangle((np.min(self.grid[:,0]), np.min(self.grid[:, 1])), np.max(self.grid[:, 0]) - np.min(self.grid[:,0]), np.max(self.grid[:, 1]) - np.min(self.grid[:, 1]), fill=False, edgecolor='red', linewidth=2))
        plt.colorbar(scatter, label="Weight")
        plt.clim(0, 1)
        plt.xlabel('Velocity', fontsize=12)
        plt.ylabel('Angle', fontsize=12)
        plt.title(title, fontsize=14)
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        plt.legend(loc='upper right')
        if save_path is not None:
            plt.savefig(save_path)  


    def add_episode_to_counts(self,vel, angle, tmp = False):
        """
        Counts to which cell the given episode belongs to by updating the counts-dict.
        Args:
            vel (float): The velocity value of the episode.
            angle (float): The angle value of the episode.
            tmp (bool): If True, updates the temporary counts instead of the main counts. Default is False.
        Returns:
            None
        """
        if tmp:
            counts = self.tmp_counts
        else:
            counts = self.counts
        vel,angle = self._get_node(vel, angle)[0]
        key = tuple((vel,angle))
        counts[key] += 1

    def reset_tmp_counts(self):
        """
        Resets the temporary counts dictionary to zero for all grid cells.
        Args:
            None
        Returns:
            None
        """
        for key in self.counts.keys():
            self.tmp_counts[key] = 0

    def plot_counts(self, tmp = False, save_path = None):
        """
        Plots the number of episodes in each grid cell as a table.
        Args:
            tmp (bool): If True, uses the temporary counts instead of the main counts. Default is False.
            save_path (str): The path to save the plot. If None, the plot is not saved.
        Returns:
            None
        """
        if tmp:
            counts = self.tmp_counts
        else:
            counts = self.counts
        # recover centers
        vel_centers   = np.unique(self.grid[:,0])
        angle_centers = np.unique(self.grid[:,1])
        n_vel, n_ang  = len(vel_centers), len(angle_centers)

        # build a 2D array from the dict counts
        counts_2d = np.zeros((n_ang, n_vel), dtype=int)
        for i, a in enumerate(np.flip(angle_centers)):
            for j, v in enumerate(vel_centers):
                counts_2d[i, j] = counts[(v, a)]

        # labels
        col_labels = [f"{v:.2f}" for v in vel_centers]
        row_labels = [f"{a:.2f}" for a in np.flip(angle_centers)]

        # render as a table
        _, ax = plt.subplots(figsize=(n_vel*1.5 + 2, n_ang*0.4 + 2))
        ax.axis('off')
        tbl = ax.table(
            cellText=counts_2d,
            rowLabels=row_labels,
            colLabels=col_labels,
            cellLoc='center',
            loc='center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)

        # shade every second data row and highlight zero-angle row
        zero_row = np.where(angle_centers == 0)[0][0] + 1  # +1 for header offset
        for row in range(1, n_ang+1):
            if row % 2 == 0:
                for col in range(n_vel):
                    tbl[(row, col)].set_facecolor("#FFFF00")  # yellow
            if row == zero_row:
                for col in range(n_vel):
                    tbl[(row, col)].set_facecolor("#FF0000")  # red

        plt.title("Exact sample counts per grid cell", pad=20)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)