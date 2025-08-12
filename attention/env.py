import numpy as np
from dataclasses import dataclass
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from ml4co_kit import BaseEnv


class TSPDataset(Dataset):
    def __init__(self, file_path: str):
        # read the data form .txt
        with open(file_path, "r") as file:
            points_list = list()
            tour_list = list()
            for line in file:
                line = line.strip()
                split_line = line.split(" output ")
                # parse points
                points = split_line[0].split(" ")
                points = np.array([[float(points[i]), float(points[i + 1])] 
                                   for i in range(0, len(points), 2)])
                points_list.append(points)
                # parse tour
                tour = split_line[1].split(" ")
                tour = np.array([int(t) for t in tour])
                tour -= 1  # convert to 0-based index
                tour_list.append(tour)
        self.points = np.array(points_list)
        self.tours = np.array(tour_list)
        
        # Convert to tensors
        self.points = torch.tensor(self.points, dtype=torch.float32)  # Shape: (num_samples, num_nodes, 2)
        self.tours = torch.tensor(self.tours, dtype=torch.long)  # Shape: (num_samples, num_nodes + 1)

    def __getitem__(self, index):
        return self.points[index], self.tours[index]  # Shape: (V, 2) and (V+1,)
    
    def __len__(self):
        return self.points.shape[0]  # number of samples
    

@dataclass
class StepState:
    """
    A data class to hold the state of the environment at each decoding step.
    This makes passing state information to the model cleaner.
    """
    current_node: Tensor = None  # Shape: (batch,)
    tours: Tensor = None  # Shape: (batch, time_step)
    mask: Tensor = None  # Shape: (batch, num_nodes)
    

class AttentionEnv(BaseEnv):
    def __init__(
        self,
        mode: str = "train",
        train_batch_size: int = 4,
        val_batch_size: int = 4,
        train_path: str = None,
        val_path: str = None,
        num_workers: int = 4,
        device: str = "cpu",
    ):
        super(AttentionEnv, self).__init__(
            name="GNNEnv",
            mode=mode,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            train_path=train_path,
            val_path=val_path,
            num_workers=num_workers,
            device=device
        )
        if mode is not None:
            self.load_data()
        self.num_nodes = self.train_dataset.points.shape[1] if self.train_dataset else None
            
        self.points = None
        self.batch_size = None
        # These will be managed during reset and step
        self.current_node = None
        self.tours = None
        self.mask = None

    def load_data(self):
        self.train_dataset = TSPDataset(self.train_path) if self.train_path else None
        self.val_dataset = TSPDataset(self.val_path) if self.val_path else None
        
    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=self.train_batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset, 
            batch_size=self.val_batch_size, 
            shuffle=False
        )
        return val_dataloader
    
    def reset(self, points: Tensor):
        """
        Resets the environment for a new rollout.
        """
        self.points = points.to(self.device)  # Shape: (batch_size, num_nodes, 2)
        self.batch_size = self.points.size(0)
        self.current_node = None
        self.tours = torch.zeros((self.batch_size, 0), dtype=torch.long, device=self.device)
        self.mask = torch.ones((self.batch_size, self.num_nodes), device=self.device)
        state_step = StepState(current_node=self.current_node, tours=self.tours, mask=self.mask)
        return state_step, None, None  # Initial state, no reward, not done

    def step(self, selected_node: Tensor):
        """
        Updates the environment state based on the selected node.
        Args:
            selected_node (Tensor): The node selected by the policy model.
                                    Shape: (batch_size,).
        Returns:
            A tuple containing:
            - state (StepState): The new state of the environment.
            - reward (Tensor or None): The final reward (negative tour length) if done, else None.
            - done (bool): A boolean indicating if the tour is complete.
        """
        self.current_node = selected_node
        self.tours = torch.cat([self.tours, self.current_node.unsqueeze(-1)], dim=1)
        self.mask.scatter_(dim=1, index=self.current_node.unsqueeze(-1), value=0)  # Mark the selected node as visited
        
        done = (self.tours.size(1) == self.num_nodes)
        reward = -self.evaluate() if done else None  # Negative tour length as reward
        state_step = StepState(current_node=self.current_node, tours=self.tours, mask=self.mask)
        return state_step, reward, done
        
    def evaluate(self):
        """
        Calculates the total length of the generated tours.

        Returns:
            Tensor: The total length for each tour in the batch. Shape: (batch_size,).
        """
        # Gather coordinates in tour order.
        # self.tours.shape: (batch_size, num_nodes)
        tour_coords = torch.gather(input=self.points, dim=1, index=self.tours.unsqueeze(-1).expand(-1, -1, 2))  # Shape: (batch_size, num_nodes, 2)
        
        # Calculate distances between consecutive nodes, including returning to the start
        rolled_coords = tour_coords.roll(dims=1, shifts=-1)
        segment_lengths = torch.norm(tour_coords - rolled_coords, dim=2)
        
        return segment_lengths.sum(dim=1)
