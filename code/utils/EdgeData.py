import numpy as np
import torch


class EdgeData:
    def __init__(self, feature_labels, max_length=float("inf")):
        # self.edge_index = np.array([[], []])
        self.edge_index = None
        # self.edge_attr = np.array([])
        self.edge_attr = None
        # self.edge_labels = np.array([])
        self.edge_labels = None
        self.feature_labels = feature_labels
        # Seeing weird results where distances are off by like 0.00001.
        # Not sure why but adding a little padding in the max length for hacky fix.
        self.max_length = max_length + 0.1

    # def are_adjacent(self, n1, n2):
    #     source = self.edge_index[0]
    #     sink = self.edge_index[1]

    #     incident_edges = np.where(source == n1)

    #     if n2 in sink[incident_edges]:
    #         return True
    #     else:
    #         return False

    def get_adjacent_nodes(self, node_index):
        todo = True
        # adjacent_node_indices = torch.tensor([])

        # for node_index in node_indices:
        #     incident_edge_indices = torch.where(edge_index[0] == node_index)[0]
        #     adjacent_node_indices = torch.hstack((adjacent_node_indices, edge_index[1][incident_edge_indices]))

        # return torch.unique(adjacent_node_indices).long()

    def add_edge(self, n1, n2, distance, label):
        if distance > self.max_length:
            return None

        # tail = np.array([[n1, n2], [n2, n1]])
        tail = torch.tensor([[n1, n2], [n2, n1]]).int()
        vector_multiplier = 2

        if n1 == n2:
            # tail = np.array([[n1], [n2]])
            tail = torch.tensor([[n1], [n2]]).int()
            vector_multiplier = 1

        # label_vector = np.zeros(len(self.feature_labels))
        label_vector = torch.zeros(len(self.feature_labels)).int()
        label_vector[self.feature_labels.index(label)] = 1
        label_vector = torch.vstack([label_vector] * vector_multiplier).int()

        attr_vector = torch.tensor([distance] * vector_multiplier).float()

        if self.edge_index is not None:
            self.edge_index = torch.hstack((self.edge_index, tail))
        else:
            self.edge_index = tail

        if self.edge_attr is not None:
            self.edge_attr = torch.hstack((self.edge_attr, attr_vector))
        else:
            self.edge_attr = attr_vector

        if self.edge_labels is not None:
            self.edge_labels = torch.vstack((self.edge_labels, label_vector))
        else:
            self.edge_labels = label_vector

    def get_data(self):
        return self.edge_index, self.edge_attr, self.edge_labels
