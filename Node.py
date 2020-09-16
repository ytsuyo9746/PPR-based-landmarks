import random

class Node():
    def __init__(self, name, adjacent_list, Graph):
        self.id = name
        self.degree = len(adjacent_list)
        self.adj = adjacent_list
        self.graph = Graph

    def __repr__(self):
        return self.id

    def __str__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    def get_random_adjacent(self):
        if self.degree == 0:
            return self
        else:
            return random.choice(self.adj)

    def get_adjacent_list(self):
        if self.degree != 0:
            return self.adj
        else:
            return [self]

    def get_adjacent_ids(self):
        adj_ids = list()
        for node in self.adj:
            adj_ids.append(node.id)
        return adj_ids

    def replace_adjacent_list(self, nodes_list):
        self.adj = nodes_list
        return

class NodeForForwardPush(Node):
    def __init__(self, name, adjacent_list, Graph):
        super().__init__(name, adjacent_list, Graph)
        self.set_P(0)
        self.set_R(0)

    def add_P(self, diff):
        self.set_P(self.P + diff)

    def add_R(self, diff):
        self.set_R(self.R + diff)

    def set_P(self, val):
        self.P = val

    def set_R(self, val):
        self.R = val
        self.set_avg_R()

    def set_avg_R(self):
        if self.degree != 0:
            self.avg_R = self.R / self.degree
        else:
            self.avg_R = self.R

    def fwd_push(self, alpha):
        self.add_P(alpha * self.R)
        R_diff = (1 - alpha) * self.avg_R
        self.set_R(0)
        for adj_node in self.get_adjacent_list():
            adj_node.add_R(R_diff)
        return

class NodeForRandomWalk(Node):
    def __init__(self, name, adjacent_list, Graph):
        super().__init__(name, adjacent_list, Graph)
        self.score = 0

    def add_score(self, diff):
        self.set_score(self.score + diff)

    def set_score(self, val):
        self.score = val
