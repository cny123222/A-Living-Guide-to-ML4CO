from torch import Tensor
from ml4co_kit import np_sparse_to_dense
import numpy as np

class GNNDecoder():
    def __init__(self, decoding_type: str = "greedy"):
        self.decoding_type = decoding_type
        
    def decode(self, heatmap: Tensor, nodes_num: int, edge_index: Tensor):
        """
        Args:
            heatmap: (E,) tensor representing edges being selected
            nodes_num: int, number of nodes
            edge_index: (2, E) Tensor with edges representing connections from source to target nodes
        Returns:
            tour: (V,) tensor representing the tour
        """
        heatmap = np.array(heatmap)
        heatmap = np_sparse_to_dense(
            nodes_num=nodes_num, edge_index=np.array(edge_index), edge_attr=heatmap
        )  # Convert into a real heatmap (V, V)
        if self.decoding_type == "greedy":
            return self._greedy_decode(heatmap)
        else:
            raise NotImplementedError(f"Decoding type '{self.decoding_type}' is not supported.")

    def _greedy_decode(self, heatmap: np.ndarray):
        batch_size = heatmap.size(0)
        num_nodes = heatmap.size(1)
        tours = []
        # Iterate over each instance
        for idx in range(batch_size):
            tour = []
            current = None
            for _ in range(num_nodes):
                if current is None:
                    # Start from the first node
                    next_node = 0
                else:
                    # Select the next node with the highest probability
                    next_node = np.argmax(heatmap[idx][current]).item()
                tour.append(next_node)
                heatmap[idx][:, next_node] = 0  # Remove the selected node
                current = next_node
        tours.append(np.array(tour))
        return np.array(tours)