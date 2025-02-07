import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


class GridAdaptiveCurriculum:
    def __init__(self, vel_range=(-1.0, 1.0), angle_range=(-1, 1), resolution=(0.5, 0.5), success_threshold=1000):
        self.resolution_vel = resolution[0]
        self.resolution_angle = resolution[1]
        self.success_threshold = success_threshold

        self._grid = self._create_grid(vel_range, angle_range, self.resolution_vel, self.resolution_angle)

    @property
    def weights(self):
        return self._grid[:, 2]
    
    @property
    def grid(self):
        return self._grid[:, :2]
    
    def set_success_threshold(self, success_threshold):
        self.success_threshold = success_threshold
    
    def _create_grid(self, vel_range, angle_range, resolution_vel, resolution_angle):
        """
        Erstellt ein Grid mit Velocity und Winkelwerten innerhalb der gegebenen Bereiche.

        Parameter:
            vel_range (tuple): Der Bereich der Velocity-Werte (min, max).
            angle_range (tuple): Der Bereich der Winkelwerte (min, max).
            resolution (float): Der Abstand zwischen den Punkten entlang der Achsen.

        Returns:
            numpy.ndarray: Das erstellte Grid als 2D-Array (n, 2).
        """
        vel_values = np.arange(vel_range[0], vel_range[1] + resolution_vel/10, resolution_vel)
        angle_values = np.arange(angle_range[0], angle_range[1] + resolution_angle/10, resolution_angle)
        # grid = np.array([[vel, angle, 1] for vel in vel_values for angle in angle_values])                ### weights are initialized with 1 and then normalized
        # grid[:, 2] = 1/np.sum(grid[:, 2])                                                                 ### weights are initialized with 1 and then normalized
        grid = np.array([[vel, angle, 0] for vel in vel_values for angle in angle_values])                  ### weights are initialized with 0

        return grid
    
    def _extend_grid_velocity(self):
        """
        Erweitert ein bestehendes Grid entlang der Velocity-Achse um eine bestimmte Anzahl von Schritten.

        Returns:
            numpy.ndarray: Das erweiterte Grid in Row-Major Order mit aktualisierten Indizes.
        """
        # Extrahieren der maximalen Velocity aus dem bestehenden Grid
        max_velocity = np.max(self.grid[:, 0])
        angle_values = np.unique(self.grid[:, 1])
        
        # Erstellen der neuen Punkte für die Erweiterung
        max_velocity += self.resolution_vel
        # new_points = np.array([[max_velocity, a, np.min(self.weights)] for a in angle_values])     ### weights are initialized with 1 and then normalized
        new_points = np.array([[max_velocity, a, 0] for a in angle_values])                          ### weights are initialized with 0

        # Erweitern des Grids um die neuen Punkte
        self._grid = np.row_stack((self._grid[:, :], new_points))
        # self._grid[:, 2] = self._grid[:, 2] / np.sum(self.weights)                                 ### weights are initialized with 1 and then normalized

    def _extend_grid_angle_top(self):
        """
        Erweitert ein bestehendes Grid entlang der Winkel-Achse (y-Achse) um eine bestimmte Anzahl von Schritten.

        Returns:
            numpy.ndarray: Das erweiterte Grid in Row-Major Order als 2D-Array (n, 3).
        """
        # Extrahieren des maximalen und minimalen Winkels aus dem bestehenden Grid
        max_angle = np.max(self.grid[:, 1])
        velocity_values = np.unique(self.grid[:, 0])
        angle_values = np.unique(self.grid[:, 1])

        # Erstellen der neuen Punkte für die Erweiterung
        max_angle += self.resolution_angle
        # new_points = np.array([[v, max_angle, np.min(self.weights)] for v in velocity_values])    ### weights are initialized with 1 and then normalized
        new_points = np.array([[v, max_angle, 0] for v in velocity_values])                          ### weights are initialized with 0

        new_point_indices = np.arange(len(angle_values), len(self.grid) + len(angle_values), len(angle_values))

        # Erweitern des Grids um die neuen Punkte
        self._grid = np.insert(self._grid, new_point_indices, new_points, axis=0)  
        # self._grid[:, 2] = self._grid[:, 2] / np.sum(self.weights)                                ### weights are initialized with 1 and then normalized

    def _extend_grid_angle_bottom(self):
        """
        Erweitert ein bestehendes Grid entlang der Winkel-Achse (y-Achse) um eine bestimmte Anzahl von Schritten.

        Returns:
            numpy.ndarray: Das erweiterte Grid in Row-Major Order als 2D-Array (n, 3).
        """
        # Extrahieren des maximalen und minimalen Winkels aus dem bestehenden Grid
        min_angle = np.min(self.grid[:, 1])
        velocity_values = np.unique(self.grid[:, 0])
        angle_values = np.unique(self.grid[:, 1])

        # Erstellen der neuen Punkte für die Erweiterung
        min_angle -= self.resolution_angle
        # new_points = np.array([[v, min_angle, np.min(self.weights)] for v in velocity_values])    ### weights are initialized with 1 and then normalized
        new_points = np.array([[v, min_angle, 0] for v in velocity_values])                          ### weights are initialized with 0

        new_point_indices = np.arange(0, len(self.grid), len(angle_values))

        # Erweitern des Grids um die neuen Punkte
        self._grid = np.insert(self._grid, new_point_indices, new_points, axis=0)
        # self._grid[:, 2] = self._grid[:, 2] / np.sum(self.weights)                                ### weights are initialized with 1 and then normalized
    
    def _get_adjacents(self, index):
        resolution = np.array([self.resolution_vel, self.resolution_angle])
        adjacent_inds = np.logical_and(
            self.grid[:, :] >= self.grid[index, :] - 1.5*resolution,
            self.grid[:, :] <= self.grid[index, :] + 1.5*resolution
        ).all(axis=1)

        return adjacent_inds
    
    def _adapt_weights_new(self, index):
        ### weights are initializd with 0 + weights are adapted diffently: instead of adding 0.2, the weights are set to 1 for the node itself and for all adjacent nodes
        self._grid[index, 2] = 1
        adjacents = self._get_adjacents(index)
        adjacent_inds = np.array(adjacents.nonzero()[0])
        self._grid[adjacent_inds, 2] = 1

    def _adapt_weights(self, index):
        self._grid[index, 2] = np.clip(self._grid[index, 2] + 0.2, 0, 1)
        # self._grid[:,2] = self._grid[:,2] / np.sum(self.weights)                          ### weights are initialized with 1 and then normalized
        adjacents = self._get_adjacents(index)
        adjacent_inds = np.array(adjacents.nonzero()[0])
        self._grid[adjacent_inds, 2] = np.clip(self._grid[adjacent_inds, 2] + 0.2, 0, 1)
        # self._grid[:,2] = self._grid[:,2] / np.sum(self.weights)  

    def _get_node(self, velocity, angle):
        # Find the closest grid point to the given velocity and angle
        squared_distances = (self.grid[:, 0] - velocity) ** 2 + (self.grid[:, 1] - angle) ** 2
        index = np.argmin(squared_distances)
        return self.grid[index], index
    
    def _is_border_vel(self, index):
        max_vel = np.max(self.grid[:, 0])
        if self.grid[index, 0] == max_vel:
            return True
        return False
    
    def _is_border_angle_top(self, index):
        max_angle = np.max(self.grid[:, 1])
        if self.grid[index, 1] == max_angle:
            return True
        return False

    def _is_border_angle_bottom(self, index):
        min_angle = np.min(self.grid[:, 1])
        if self.grid[index, 1] == min_angle:
            return True
        return False

    def update(self, velocity, angle, reward):
        if reward >= self.success_threshold:
            _, node_idx = self._get_node(velocity, angle)
            if self._is_border_vel(node_idx) and self.grid[node_idx, 0] < 1.25:
                self._extend_grid_velocity()
            if self._is_border_angle_top(node_idx) and self.grid[node_idx, 1] < np.pi:
                self._extend_grid_angle_top()
            if self._is_border_angle_bottom(node_idx) and self.grid[node_idx, 1] > -np.pi:
                self._extend_grid_angle_bottom()
            _, node_idx = self._get_node(velocity, angle)
            self._adapt_weights(node_idx)
    
    def _sample_node(self):
        """default to uniform"""
        if self.weights.sum() == 0:                                                              ### weights are initialized with 0
            index = np.random.choice(len(self.grid), 1)                                           ### weights are initialized with 0
        else:                                                                                       ### weights are initialized with 0
            index = np.random.choice(len(self.grid), 1, p=self.weights / self.weights.sum())          ### weights are initialized with 0
        # index = np.random.choice(len(self.grid), 1, p=self.weights)                            ### weights are initialized with 1 and then normalized
        return self.grid[index][0], index[0]

    def _sample_uniform_from_cell(self, center):
        cell_sizes = np.array([self.resolution_vel, self.resolution_angle])
        low, high = center + cell_sizes / 2, center - cell_sizes / 2
        return np.random.uniform(low, high)

    def sample(self):
        center, index = self._sample_node()
        # return self._sample_uniform_from_cell(center), index
        return center, index
    
    def plot(self, title, label, save_path=None):
        fig = plt.figure()
        # plt.scatter(self.grid[:,0], self.grid[:,1],s=25, c = self.weights, cmap='viridis')
        scatter = plt.scatter(self.grid[:,0], self.grid[:,1], s=self.weights*100, c=self.weights, cmap='viridis', alpha=1, edgecolors='w', label = label)
        plt.gca().add_patch(plt.Rectangle((np.min(self.grid[:,0]), np.min(self.grid[:, 1])), np.max(self.grid[:, 0]), np.max(self.grid[:, 1]) - np.min(self.grid[:, 1]), fill=False, edgecolor='red', linewidth=2))
        plt.colorbar(scatter, label="Weight")
        plt.clim(0, 1)
        # plt.xlim(0,1.25)
        # plt.ylim(-np.pi,np.pi)
        plt.xlabel('Velocity', fontsize=12)
        plt.ylabel('Angle', fontsize=12)
        plt.title(title, fontsize=14)
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        plt.legend(loc='upper right')
        if save_path is not None:
            plt.savefig(save_path)  