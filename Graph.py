from Node import *
import random
from queue import Queue
import pdb
import numpy as np

class Graph():
    def __init__(self, ADJ, mode='forward_local_push'):
        self.nodes = dict()
        self.mode = mode
        if mode == 'forward_local_push':
            for node, adjacent_list in ADJ.items():
                self.nodes[node] = NodeForForwardPush(node, adjacent_list, self)
            for adjacent_list in ADJ.values():
                for node_id in adjacent_list:
                    if node_id not in self.nodes.keys():
                        self.nodes[node_id] = NodeForForwardPush(node_id, [node_id], self)

        elif mode == 'random_walk':
            for node, adjacent_list in ADJ.items():
                self.nodes[node] = NodeForRandomWalk(node, adjacent_list, self)
            for adjacent_list in ADJ.values():
                for node_id in adjacent_list:
                    if node_id not in self.nodes.keys():
                        self.nodes[node_id] = NodeForRandomWalk(node_id, [node_id], self)

        for node in self.nodes.values():
            adjacent_list = node.get_adjacent_list()
            nodes_list = list()
            for node_id in adjacent_list:
                nodes_list.append(self.search_node(node_id))
            node.replace_adjacent_list(nodes_list)
        self.node_count = len(self.nodes)

    def __repr__(self):
        adj = dict()
        for node in self.nodes.values():
            adj[str(node)] = set()
            for adj_node in node.adj:
                adj[str(node)].add(str(adj_node))
        return str(adj)

    def __eq__(self, other):
        if set(self.nodes.keys()) != set(other.nodes.keys()):
            return False
        else:
            for node_id in self.nodes.keys():
                if set(self.nodes[node_id].get_adjacent_ids()) != \
                set(other.nodes[node_id].get_adjacent_ids()):
                    return False
            return True

    def get_dst_dict_from_source(self, source_node):
        dst = dict()
        Q = Queue()
        Q.put(source_node)
        dst = {source_node.id: 0}

        while not Q.empty():
            node = Q.get()
            for adj_node in node.adj:
                if adj_node.id not in dst.keys():
                    dst[adj_node.id] = dst[node.id] + 1
                    Q.put(adj_node)
                else:
                    pass
        return dst

    def get_dst_between(self, source_node, destination_node):
        if source_node == destination_node:
            return 0
        dst = dict()
        Q = Queue()
        Q.put(source_node)
        dst = {source_node.id: 0}
        destination_id = destination_node.id

        while not Q.empty():
            node = Q.get()
            for adj_node in node.adj:
                if adj_node.id not in dst.keys():
                    if adj_node.id == destination_id:
                        return dst[node.id] + 1
                    dst[adj_node.id] = dst[node.id] + 1
                    Q.put(adj_node)
                else:
                    pass
        return None

    def calc_PPR_by_power_iteration(self, source_id, alpha, epsilon):
        transition_matrix = self.create_transiton_matrix_for_PPR(source_id, alpha)
        ppr_vec = np.random.rand(self.node_count)
        norm = np.linalg.norm(ppr_vec,ord=1)
        for i in range(self.node_count):
            ppr_vec[i] /= norm

        prev = ppr_vec
        new = np.dot(transition_matrix, prev)
        count = 1
        diff = 1
        while (diff > epsilon):
            prev = new
            new = np.dot(transition_matrix, prev)
            diff = np.linalg.norm((new - prev), ord=1)
            # print('count : {}, diff : {}'.format(count, diff))
            count += 1
        return new

    def create_transiton_matrix_for_PPR(self, source_id, alpha):
        node_id_list = list(self.nodes.keys())
        node_id_list.sort()
        node_id_to_index = dict()
        for i in range(len(node_id_list)):
            node_id_to_index[node_id_list[i]] = i

        transition_matrix = np.zeros((self.node_count, self.node_count))
        for i, node_id in enumerate(node_id_list):
            for adj_node in self.nodes[node_id].adj:
                transition_matrix[node_id_to_index[adj_node.id]][i] = 1

        dangling = np.max(transition_matrix, axis = 0)
        for i in range(self.node_count):
            if dangling[i] == 0:
                dangling[i] = 1
            else:
                dangling[i] = 0
        transition_matrix += dangling

        count = np.count_nonzero(transition_matrix, axis = 0)
        transition_matrix = transition_matrix / count

        source_index = node_id_to_index[source_id]
        pref_vec = np.zeros((self.node_count, 1))
        pref_vec[source_index][0] = 1
        transition_matrix = (1 - alpha) * transition_matrix + alpha * pref_vec

        return transition_matrix

    def get_random_node(self):
        return random.choice(nodes.values())

    def search_node(self, node_id):
        if node_id in self.nodes.keys():
            return self.nodes[node_id]
        else:
            return None

    def get_subgraph(self, source_node, hop, mode='forward_local_push'):
        ADJ_subgraph = dict()
        Q = Queue()
        Q.put(source_node)
        dst = {source_node.id: 0}

        while not Q.empty():
            node = Q.get()
            ADJ_subgraph[node.id] = list()
            for adj_node in node.adj:
                if adj_node.id not in dst.keys():
                    dst[adj_node.id] = dst[node.id] + 1
                    if dst[adj_node.id] <= hop:
                        Q.put(adj_node)
                else:
                    pass
                if dst[adj_node.id] <= hop:
                    ADJ_subgraph[node.id].append(adj_node.id)

        subgraph = Graph(ADJ_subgraph, mode)

        return subgraph

    def get_max_avgR_node(self):
        if self.mode == 'forward_local_push':
            return self.nodes[max(self.nodes, key=lambda x:self.nodes[x].avg_R)]
        else: return None

    def get_min_avgR_node(self):
        if self.mode == 'forward_local_push':
            return self.nodes[min(self.nodes, key=lambda x:self.nodes[x].avg_R)]
        else: return None

    def set_all_PandR_0(self):
        if self.mode == 'forward_local_push':
            for node in self.nodes.values():
                node.set_P(0)
                node.set_R(0)
        else: return None

    def set_all_R(self, val):
        if self.mode == 'forward_local_push':
            for node in self.nodes.values():
                node.set_R(val)
        else: return None

    def set_all_score_0(self):
        if self.mode == 'random_walk':
            for node in self.nodes.values():
                node.set_score(0)
        else: return None
