import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


class GridAdaptiveCurriculum:
    def __init__(self, vel_range=(-1.0, 1.0), angle_range=(-1, 1), resolution=(0.5, 0.5), success_threshold=1000, seed = None):
        self.resolution_vel = resolution[0]
        self.resolution_angle = resolution[1]
        self.success_threshold = success_threshold

        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

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
        vel_values = np.arange(vel_range[0], vel_range[1] + resolution_vel, resolution_vel)
        angle_values = np.arange(angle_range[0], angle_range[1] + resolution_angle, resolution_angle)
        grid = np.array([[vel, angle, 0] for vel in vel_values for angle in angle_values])      ###alternativ: eventuell weights als 1 definieren und dann direkt normalisieren
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
        new_points = np.array([[max_velocity, a, 0] for a in angle_values])

        # Erweitern des Grids um die neuen Punkte
        self._grid = np.row_stack((self._grid[:, :], new_points))

    def _extend_grid_angle(self):
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
        new_points = np.array([[v, max_angle, 0] for v in velocity_values])
        new_point_indices = np.arange(len(angle_values), len(self.grid) + len(angle_values), len(angle_values))

        # Erweitern des Grids um die neuen Punkte
        self._grid = np.insert(self._grid, new_point_indices, new_points, axis=0)           
    
    def _get_adjacents(self, index):
        resolution = np.array([self.resolution_vel, self.resolution_angle])
        adjacent_inds = np.logical_and(
            self.grid[:, :] >= self.grid[index, :] - 1.5*resolution,
            self.grid[:, :] <= self.grid[index, :] + 1.5*resolution
        ).all(axis=1)

        return adjacent_inds
    
    def _adapt_weights(self, index):
        self._grid[index, 2] = np.clip(self._grid[index, 2] + 0.2, 0, 1)
        adjacents = self._get_adjacents(index)
        adjacent_inds = np.array(adjacents.nonzero()[0])
        self._grid[adjacent_inds, 2] = np.clip(self._grid[adjacent_inds, 2] + 0.2, 0, 1)

    def _get_node(self, velocity, angle):
        # Find the closest grid point to the given velocity and angle
        squared_distances = (self.grid[:, 0] - velocity) ** 2 + (self.grid[:, 1] - angle)
        index = np.argmin(squared_distances)
        return self.grid[index], index
    
    def _is_border_vel(self, index):
        max_vel = np.max(self.grid[:, 0])
        if self.grid[index, 0] == max_vel:
            return True
        return False
    
    def _is_border_angle(self, index):
        max_angle = np.max(self.grid[:, 1])
        if self.grid[index, 1] == max_angle:
            return True
        return False

    def update(self, velocity, angle, reward):
        if reward >= self.success_threshold:
            _, node_idx = self._get_node(velocity, angle)
            if self._is_border_vel(node_idx):
                self._extend_grid_velocity()
            if self._is_border_angle(node_idx):
                self._extend_grid_angle()
            self._adapt_weights(node_idx)
    
    def _sample_node(self):
        """default to uniform"""
        if self.weights.sum() == 0:
            index = self.rng.choice(len(self.grid), 1)
        else:
            index = self.rng.choice(len(self.grid), 1, p=self.weights / self.weights.sum())
        return self.grid[index][0], index[0]

    def _sample_uniform_from_cell(self, center):
        cell_sizes = np.array([self.resolution_vel, self.resolution_angle])
        low, high = center + cell_sizes / 2, center - cell_sizes / 2
        return self.rng.uniform(low, high)

    def sample(self):
        center, index = self._sample_node()
        return self._sample_uniform_from_cell(center), index
    
    def plot(self, title, label, save_path=None):
        fig = plt.figure()
        plt.scatter(self.grid[:,0], self.grid[:,1], label=label,s=25, c = self.weights, cmap='viridis')
        scatter = plt.scatter(self.grid[:,0], self.grid[:,1], s=self.weights*750, c=self.weights, cmap='viridis', alpha=1, edgecolors='w')
        plt.colorbar(scatter, label="Weight")
        plt.xlabel('Velocity', fontsize=12)
        plt.ylabel('Angle', fontsize=12)
        plt.title(title, fontsize=14)
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        plt.legend(loc='upper right')
        if save_path is not None:
            plt.savefig(save_path)  
    

# if __name__ == '__main__':
#     fig = plt.figure()

#     gridAdaptiveCurric = GridAdaptiveCurriculum(vel_range=(0.0, 2.0), angle_range=(-1.0, 1.0), resolution =(0.5, 0.5))
    
#     grid = gridAdaptiveCurric.grid
#     weights = gridAdaptiveCurric.weights
#     print("Initial grid shape: ", grid.shape)
#     print("Initial weights: ", weights)
#     # plt.scatter(grid[:, 0], grid[:, 1], label='Grid Punkte',s=200)
#     print("--------------------------------------------------------------------")

#     sample,index=gridAdaptiveCurric.sample()
#     print("sample",sample, " in cell of node with index ", index)
#     print("--------------------------------------------------------------------")

#     gridAdaptiveCurric.update(1.1, 0.7, 600)
#     extendedGrid = gridAdaptiveCurric.grid
#     weights = gridAdaptiveCurric.weights
#     print("Updated grid shape: ", extendedGrid.shape)
#     print("Updated weights: ", weights)
#     print("--------------------------------------------------------------------")

#     gridAdaptiveCurric.update(1.1, 0.7, 1000)
#     extendedGrid = gridAdaptiveCurric.grid
#     weights = gridAdaptiveCurric.weights
#     print("Updated grid shape: ", extendedGrid.shape)
#     print("Updated weights: ", weights)
#     # plt.scatter(extendedGrid[:, 0], extendedGrid[:, 1], label='exGrid Punkte',s=100)
#     print("--------------------------------------------------------------------")

#     gridAdaptiveCurric.update(1.9, 0.61, 1500)
#     extendedGrid2 = gridAdaptiveCurric.grid
#     x = extendedGrid2[:,0]
#     y = extendedGrid2[:,1]
#     weights = gridAdaptiveCurric.weights
#     print("Updated grid shape: ", extendedGrid2.shape)
#     print("Updated weights: ", weights)
#     plt.scatter(x, y, label='exGrid2 Punkte',s=25, c = weights, cmap='viridis')
#     plots = ["hexbin", "scatter", "contourf", "weighted histogram", "vector field", None]
#     plot = plots[1]
#     if plot == "hexbin":
#         #first way of plotting: hexbin
#         plt.colorbar(plt.hexbin(x, y, C=weights, gridsize=30, cmap='viridis', reduce_C_function=np.sum, label='exGrid2 Punkte'))
#     if plot == "scatter":
#         #second way of plotting: just scatter
#         scatter = plt.scatter(x, y, s=weights*750, c=weights, cmap='viridis', alpha=1, edgecolors='w')
#         plt.colorbar(scatter, label="Weight")
#     if plot == "contourf":
#         #third way of plotting: contourf
#         # Interpolate the data onto a grid
#         xi = np.linspace(min(x), max(x), 100)
#         yi = np.linspace(min(y), max(y), 100)
#         xi, yi = np.meshgrid(xi, yi)
#         zi = griddata((x, y), weights, (xi, yi), method='cubic')
#         # plot the grid
#         contour = plt.contourf(xi, yi, zi, levels=15, cmap='viridis')
#         plt.colorbar(contour, label='Weight')
#         plt.scatter(x, y, c='white', s=10, alpha=0.5, label='Points')
#     if plot == "weighted histogram":
#         #fourth way of plotting: weighted histogram
#         plt.hist2d(x, y, bins=30, weights=weights, cmap='viridis')
#         plt.colorbar(label='Weight')
#     if plot == "vector field":
#         #fifth way of plotting: vector field
#         plt.quiver(x, y, weights, weights, angles='xy', scale_units='xy', scale=1, cmap='viridis', label='exGrid2 Punkte')
#         plt.colorbar(label='Weight')    
    
#     plt.xlabel('Velocity', fontsize=12)
#     plt.ylabel('Winkel', fontsize=12)
#     plt.title('Grid Darstellung', fontsize=14)
#     plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
#     plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
#     plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
#     plt.legend(loc='upper right')
#     plt.show()