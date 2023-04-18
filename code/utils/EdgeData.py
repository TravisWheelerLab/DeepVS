import numpy as np

class EdgeData:
    def __init__(self, feature_labels):
        self.edge_index = np.array([[], []])
        self.edge_attr = np.array([])
        self.edge_labels = np.array([])
        self.feature_labels = feature_labels

    def are_adjacent(self, n1, n2):
        source = self.edge_index[0]
        sink = self.edge_index[1]

        incident_edges = np.where(source == n1) 

        if n2 in sink[incident_edges]:
            return True
        else:
            return False

    def get_adjacent_nodes(self, node_index):
        todo=True
        # adjacent_node_indices = torch.tensor([])

        # for node_index in node_indices:
        #     incident_edge_indices = torch.where(edge_index[0] == node_index)[0]
        #     adjacent_node_indices = torch.hstack((adjacent_node_indices, edge_index[1][incident_edge_indices]))

        # return torch.unique(adjacent_node_indices).long()
     

    def add_edge(self, n1, n2, distance, label):
        tail = np.array([[n1,n2],[n2,n1]]) 
        vector_multiplier = 2

        if n1==n2:
            tail = np.array([[n1],[n2]])
            vector_multiplier = 1

        label_vector = np.zeros(len(self.feature_labels))
        label_vector[self.feature_labels.index(label)] = 1

        self.edge_index = np.hstack((self.edge_index, tail))
        self.edge_attr = np.hstack((self.edge_attr, [distance]*vector_multiplier))

        if self.edge_labels.size == 0:
            self.edge_labels = np.array([label_vector]*vector_multiplier)
        else:
            self.edge_labels = np.vstack((self.edge_labels, [label_vector]*vector_multiplier))

    def get_data(self):
        return self.edge_index, self.edge_attr, self.edge_labels