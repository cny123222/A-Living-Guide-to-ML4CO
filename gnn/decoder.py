from torch import Tensor
from ml4co_kit import np_sparse_to_dense
import numpy as np

class GNNDecoder():
    def __init__(self, decoding_type: str = "greedy"):
        self.decoding_type = decoding_type
        
    def decode(self, heatmap: Tensor, nodes_num: int, edge_index: Tensor):
        """
        Args:
            heatmap: (B, E) tensor representing edges being selected
            nodes_num: int, number of nodes
            edge_index: (B, 2, E) Tensor with edges representing connections from source to target nodes
        Returns:
            tour: (B, V+1) tensor representing the tour
        """
        # Convert to numpy for processing
        heatmap = heatmap.cpu().numpy()
        edge_index = edge_index.cpu().numpy()
        # Convert heatmap to a dense format
        batch_size = heatmap.shape[0]
        nodes_num = heatmap.shape[1]
        heatmap_dense = np.zeros((batch_size, nodes_num, nodes_num), dtype=np.float32)
        for idx in range(batch_size):
            heatmap_dense[idx] = np_sparse_to_dense(
                nodes_num=nodes_num, edge_index=edge_index[idx], edge_attr=heatmap[idx]
            )  # Convert into a real heatmap (V, V)
        # Decode the tour based on the heatmap
        if self.decoding_type == "greedy":
            return self._greedy_decode(heatmap_dense, batch_size, nodes_num)
        else:
            raise NotImplementedError(f"Decoding type '{self.decoding_type}' is not supported.")

    def _greedy_decode(self, heatmap: np.ndarray, batch_size: int, nodes_num: int):
        """
        Args:
            heatmap: (B, V, V) numpy array representing the heatmap
            batch_size: int, number of samples in the batch
            nodes_num: int, number of nodes
        Returns:
            tours: (B, V) numpy array representing the decoded tours
        """
        tours = []
        # Iterate over each instance
        for idx in range(batch_size):
            tour = []
            current = None
            for _ in range(nodes_num):
                if current is None:
                    # Start from the first node
                    next_node = 0
                else:
                    # Select the next node with the highest probability
                    next_node = np.argmax(heatmap[idx][current]).item()
                tour.append(next_node)
                heatmap[idx][:, next_node] = 0  # Remove the selected node
                current = next_node
            tour.append(0)  # Return to the starting node
            tours.append(np.array(tour))
        return np.array(tours)