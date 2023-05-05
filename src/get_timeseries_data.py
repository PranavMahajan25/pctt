import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader


def generate_random_points(num_points=10, min_limit=-5, max_limit=5):
    points = np.random.uniform(min_limit, max_limit, (num_points, 2))
    points = points[np.argsort(points[:, 0])]  # Sort points by x-coordinate
    return points

def create_2d_spline(points, num_timepoints=1000, kind='cubic'):
    t = np.arange(len(points))
    t_new = np.linspace(0, len(points) - 1, num_timepoints)

    interp_x = interp1d(t, points[:, 0], kind=kind)
    interp_y = interp1d(t, points[:, 1], kind=kind)
    
    spline_x = interp_x(t_new)
    spline_y = interp_y(t_new)
    
    return np.column_stack((spline_x, spline_y))

def generate_2d_spline_dataset(num_splines=100, num_points=10, num_timepoints=1000):
    spline_dataset = []

    for _ in range(num_splines):
        points = generate_random_points(num_points)
        spline = create_2d_spline(points, num_timepoints)
        spline_dataset.append(spline)

    return spline_dataset

# spline_dataset = generate_2d_spline_dataset()

# # Visualize the first 10 splines
# for i in range(10):
#     plt.plot(spline_dataset[i][:, 0], spline_dataset[i][:, 1])

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('First 10 2D Splines')
# plt.show()


class SplineDataset(Dataset):
    def __init__(self, num_splines=100, num_points=10, num_timepoints=1000):
        self.spline_data = generate_2d_spline_dataset(num_splines, num_points, num_timepoints)

    def __len__(self):
        return len(self.spline_data)

    def __getitem__(self, idx):
        return torch.tensor(self.spline_data[idx], dtype=torch.float32)

# # Create the SplineDataset with 100 2D splines
# spline_dataset = SplineDataset(100)

# # Create a PyTorch DataLoader with the custom SplineDataset
# spline_dataloader = DataLoader(spline_dataset, batch_size=8, shuffle=True)

# # Iterate over the DataLoader
# for batch_idx, spline_batch in enumerate(spline_dataloader):
#     print(f'Batch {batch_idx + 1}: shape {spline_batch.shape}')