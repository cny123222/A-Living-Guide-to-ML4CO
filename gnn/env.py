import numpy as np
import torch
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
        
    def __getitem__(self, index):
        points = self.points[index]  # shape: (V, 2)
        tour = self.tours[index]     # shape: (V+1, )

        node_num = points.shape[0]
        # create edge index
        src, dst = np.meshgrid(np.arange(node_num), np.arange(node_num))
        mask = (src != dst)
        src, dst = src[mask], dst[mask]  # shape: (E, )
        edge_index = np.stack([src, dst], axis=0)  # shape: (2, E)
        edge_index = torch.tensor(edge_index, dtype=torch.long)  # shape: (2, E)

        # calculate each edge's length
        edges = np.linalg.norm(points[src] - points[dst], axis=1)  # shape: (E, )
        edges = torch.tensor(edges, dtype=torch.float32)  # shape: (E, )
        
        # generate the ground truth
        gt_adj = np.zeros((node_num, node_num), dtype=bool)
        gt_adj[tour[:-1], tour[1:]] = True
        gt_adj = gt_adj | gt_adj.T  # make it undirected
        ground_truth = gt_adj[src, dst]
        ground_truth = torch.tensor(ground_truth, dtype=torch.long)  # shape: (E, )

        return points, edges, edge_index, ground_truth, tour[:-1] # return tour without the last node
    
    
class GNNEnv(BaseEnv):
    def __init__(
        self,
        mode: str = "train",
        train_batch_size: int = 4,
        val_batch_size: int = 4,
        test_batch_size: int = 4,
        train_path: str = None,
        val_path: str = None,
        test_path: str = None,
        num_workers: int = 4,
        device: str = "cpu",
    ):
        super(GNNEnv, self).__init__(
            name="GNNEnv",
            mode=mode,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            num_workers=num_workers,
            device=device
        )
        if mode is not None:
            self.load_data()
        
    def load_data(self):
        self.train_dataset = TSPDataset(self.train_path) if self.train_path else None
        self.val_dataset = TSPDataset(self.val_path) if self.val_path else None
        self.test_dataset = TSPDataset(self.test_path) if self.test_path else None
        
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
    
    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False
        )
        return test_dataloader