import unittest
from functions import *
from Graph import *

class TestPPR(unittest.TestCase):
    def setUp(self):
        self.ADJ = {'0':['1', '2'], '1':['0', '3', '4'], '2':['0', '4', '5'], \
        '3':['1', '6'], '4':['1', '2', '5', '6'], '5':['2', '4', '8'], \
        '6':['3', '4', '7'], '7':['6', '8'], '8':['5', '7']}
        self.source = '0'
        self.alpha = 0.15
        self.epsilon = 0.0001
        self.graph = Graph(self.ADJ, 'forward_local_push')

    def tearDown(self):
        pass

    def test_ppr(self):
        # P, R = calc_PPR_by_forward_local_push(self.ADJ, self.source, self.alpha, self.epsilon)
        P, R = calc_PPR_by_forward_local_push(self.graph, self.source, self.alpha, self.epsilon)
        self.assertEqual(P, \
        {'0': 0.23905047183767272, '1': 0.15818756963182004, \
        '2': 0.15610821332467192, '3': 0.06548021403691155, \
        '4': 0.13547547953577246, '5': 0.09088141644190337, \
        '6': 0.07297155190433098, '7': 0.038581915537474205, \
        '8': 0.04213327646567948}, 'PPR')

    def test_pr(self):
        P, R = calc_PR(self.graph, self.alpha, self.epsilon)
        self.assertEqual(P, \
        {'0': 0.08576482706306779, '1': 0.12313070456053728, \
        '2': 0.1207451556620548, '3': 0.08696373421821786, \
        '4': 0.15573552426304466, '5': 0.12197880920203959, \
        '6': 0.12497659268023166, '7': 0.09011184961707466, \
        '8': 0.08952486536116794}, 'PR')

    def test_exact_landmark(self):
        landmarks = decide_exact_landmark(self.graph, '0', self.alpha, self.epsilon, bro_count=2)
        self.assertEqual(sorted(landmarks), sorted(['1', '4']), 'exact landmark')

    def test_random_landmark(self):
        graph = Graph(self.ADJ, 'random_walk')
        landmarks = decide_random_landmark(graph, '0', self.alpha, 1000, bro_count=2)
        self.assertEqual(sorted(landmarks), sorted(['1', '4']), 'random landmark')

    def test_get_subgraph(self):
        source_node = self.graph.search_node('1')
        subgraph = self.graph.get_subgraph(source_node, 2)
        exact_subgraph_adj = {'0':['1', '2'], '1':['0', '3', '4'], '2':['0', '4', '5'], \
        '3':['1', '6'], '4':['1', '2', '5', '6'], '5':['2', '4'], '6':['3', '4']}
        exact_subgraph = Graph(exact_subgraph_adj)
        self.assertEqual(subgraph, exact_subgraph, 'get subgraph')

    def test_dst_of_all_nodes(self):
        source_node = self.graph.search_node('0')
        dst = self.graph.get_dst_dict_from_source(source_node)
        exact_dst = {'0': 0, '1': 1, '2': 1, '3': 2, \
        '4': 2, '5': 2, '6': 3, '7': 4, '8': 3}
        self.assertEqual(dst, exact_dst, 'dst of all nodes from source')

    def test_get_dst_between(self):
        dst = self.graph.get_dst_between(self.graph.search_node('0'), self.graph.search_node('5'))
        self.assertEqual(dst, 2, 'dst between 0 and 5')
        dst = self.graph.get_dst_between(self.graph.search_node('3'), self.graph.search_node('8'))
        self.assertEqual(dst, 3, 'dst between 3 and 8')
        dst = self.graph.get_dst_between(self.graph.search_node('6'), self.graph.search_node('6'))
        self.assertEqual(dst, 0, 'dst between 6 and 6')
