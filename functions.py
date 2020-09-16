from time import time
import datetime
import json
import pickle
import random
from queue import Queue
import copy
from Graph import *


# directory = './dataset/web-Google/'
# f = open(directory + 'adjacent_list.json')
# ADJ = json.load(f)


# ADJ = {'s' : ['a', 'h', 'e'], 'a' : ['s', 'b'], 'b' : ['a', 'c', 'f', 'g', 'h'],\
#        'c' : ['b', 'd'], 'd' : ['c', 'e'], 'e' : ['s', 'd', 'f'], \
#        'f' : ['b', 'e'], 'g' : ['b'], 'h' : ['s', 'b']}

# ADJ = {'0':['1', '2'], '1':['0', '3', '4'], '2':['0', '4', '5'], \
# '3':['1', '6'], '4':['1', '2', '5', '6'], '5':['2', '8'], \
# '6':['3', '4', '7'], '7':['6', '8'], '8':['5', '7']}

def create_adjacent_list(directory, filename, metadata, separator='\t', save_as_file=True):
    filepath = directory + filename
    f = open(filepath)
    data = f.read()
    f.close()
    data = data.split('\n')
    del data[-1]
    del data[metadata[0]:metadata[1]]
    for i in range(len(data)):
        data[i] = data[i].split(separator)

    ADJ = dict()
    loop_count = 0
    for edge in data:
        loop_count += 1
        if edge[0] not in ADJ.keys():
            ADJ[edge[0]] = list()
        ADJ[edge[0]].append(edge[1])
        if (loop_count % 100000 == 0):
            print("{} % done".format(int(100 * loop_count / len(data))))

    if save_as_file:
        with open(directory + 'adjacent_list.json', 'w') as fp:
            json.dump(ADJ, fp)
        return
    else:
        return ADJ

def create_graph_object(directory, save_as_file=False):
    f = open(directory + 'adjacent_list.json')
    ADJ = json.load(f)
    graph_flp = Graph(ADJ, 'forward_local_push')
    graph_rw = Graph(ADJ, 'random_walk')

    if save_as_file:
        with open(directory + 'graph_for_forward_local_push.pickle', 'w') as fp:
            pickle.dump(graph_flp, fp)
        with open(directory + 'graph_for_random_walk.pickle', 'w') as fp:
            pickle.dump(graph_rw, fp)
        return
    else:
        return graph_flp, graph_rw

def get_timestamp():
    dt_now = datetime.datetime.now()
    return dt_now.strftime('%y%m%d%H%M%S')

# rtn # of nodes within distance hop from source
def complex(source, hop):
    count = 0
    Q = Queue()
    Q.put(source)
    dst = {source: 0}

    while not Q.empty():
        v = Q.get()
        for node in ADJ[v]:
            if node not in dst:
                Q.put(node)
                dst[node] = dst[v] + 1
                if dst[node] < hop:
                    Q.put(node)
                if dst[node] <= hop:
                    count += 1
            else:
                pass

    #print('source : {}'.format(source))
    #print('score : {}'.format(count))

    return count

# rtn match rate of exact_landmark and random_landmark
# preferred : exact has same length as random
# required : exact cannot be empty list (division by 0 will occur)
# input : two lists
def match_rate(exact_landmark, random_landmark):
    total = len(exact_landmark)

    exact = copy.deepcopy(exact_landmark)
    random = copy.deepcopy(random_landmark)

    count = 0
    for node in exact:
        if node in random:
            count += 1
            random.remove(node)

    print('match_rate : {}'.format(count / total))

    return count / total

def print_time(msg, times):
    print('実行時間 [{}] ; {}'.format(msg, (times[-1] - times[-2])))

def create_outdegree_list():
    d_out = dict()
    for node in ADJ.keys():
        d_out[node] = len(ADJ[node])
    return d_out

def create_adjacent_considering_dangling_node(ADJ):
    rtn_adj = dict()
    all_node_list = list(ADJ.keys())
    for node, adj in ADJ.items():
        if len(adj) == 0:
            rtn_adj[node] = all_node_list
        else:
            rtn_adj[node] = adj

    return rtn_adj

def calc_PPR_by_forward_local_push(graph, source_id, alpha, epsilon):
    graph.set_all_PandR_0()
    graph.search_node(source_id).set_R(1)

    start_time = time()
    forward_local_push(graph, alpha, epsilon)
    end_time = time()
    proc_time = end_time - start_time
    print('PPR processing time: {}'.format(proc_time))

    rtn_P = dict()
    rtn_R = dict()
    for node in graph.nodes.values():
        rtn_P[node.id] = node.P
        rtn_R[node.id] = node.R

    return rtn_P, rtn_R

def forward_local_push(graph, alpha, epsilon):
    endloop_flag = 0
    while endloop_flag == 0:
        endloop_flag = 1
        max_avgR_node = graph.get_max_avgR_node()
        if max_avgR_node.avg_R > epsilon:
            max_avgR_node.fwd_push(alpha)
            endloop_flag = 0

    endloop_flag = 0
    while endloop_flag == 0:
        endloop_flag = 1
        min_avgR_node = graph.get_min_avgR_node()
        if min_avgR_node.avg_R < -epsilon:
            min_avgR_node.fwd_push(alpha)
            endloop_flag = 0
    return

def calc_PPR_by_random_walk(graph, source_id, alpha, count):
    graph.set_all_score_0()

    start_time = time()
    source_node = graph.search_node(source_id)
    for i in range(count):
        current_node = source_node
        while (random.random() > alpha):
            current_node = current_node.get_random_adjacent()
        current_node.add_score(1)
    end_time = time()
    proc_time = end_time - start_time

    score = dict()
    for node in graph.nodes.values():
        score[node.id] = node.score / count

    return score

def calc_PR(graph, alpha, epsilon):
    graph.set_all_PandR_0()
    graph.set_all_R(1/graph.node_count)

    start_time = time()
    forward_local_push(graph, alpha, epsilon)
    end_time = time()
    proc_time = end_time - start_time
    print('PR processing time: {}'.format(proc_time))

    rtn_P = dict()
    rtn_R = dict()
    for node in graph.nodes.values():
        rtn_P[node.id] = node.P
        rtn_R[node.id] = node.R

    return rtn_P, rtn_R

def decide_landmarks_from_broker(graph, broker_id, alpha_for_land=0.15, epsilon_for_land=0.0001,\
 land_count_per_bro=1, subgraph_size=2, bro_to_lands=dict()):
    landmarks = list()
    if broker_id not in bro_to_lands.keys():
        bro_to_lands[broker_id] = list()
        broker_node = graph.search_node(broker_id)
        subgraph = graph.get_subgraph(broker_node, subgraph_size)
        PR, R = calc_PR(subgraph, alpha_for_land, epsilon_for_land)
        PR_sorted = sorted(PR.items(), key=lambda x:x[1], reverse=True)
        for landmark_id, score in PR_sorted[:land_count_per_bro]:
            bro_to_lands[broker_id].append(landmark_id)
    return bro_to_lands[broker_id]

def decide_exact_landmark(graph, source_id, alpha_for_bro, epsilon_for_bro, \
alpha_for_land=0.15, epsilon_for_land=0.0001, bro_count=5, land_count_per_bro=1,\
subgraph_size=2, bro_to_lands=dict()):
    landmarks = list()

    exact_P, exact_R = calc_PPR_by_forward_local_push(graph, source_id, alpha_for_bro, epsilon_for_bro)
    exact_P_sorted = sorted(exact_P.items(), key=lambda x:x[1], reverse=True)
    for broker_id, score in exact_P_sorted[:bro_count]:
        landmarks += decide_landmarks_from_broker(graph, broker_id, alpha_for_land, epsilon_for_land,\
        land_count_per_bro, subgraph_size, bro_to_lands)
    # print('Landmark: {}'.format(landmark_id))

    return landmarks

def decide_random_landmark(graph, source_id, alpha_for_bro, random_count, \
alpha_for_land=0.15, epsilon_for_land=0.0001, bro_count=5, land_count_per_bro=1,\
subgraph_size=2, bro_to_lands=dict()):
    landmarks = list()

    random_P = calc_PPR_by_random_walk(graph, source_id, alpha_for_bro, random_count)
    random_P_sorted = sorted(random_P.items(), key=lambda x:x[1], reverse=True)
    for broker_id, score in random_P_sorted[:bro_count]:
        landmarks += decide_landmarks_from_broker(graph, broker_id, alpha_for_land, epsilon_for_land,\
        land_count_per_bro, subgraph_size, bro_to_lands)

    return landmarks

def calc_bc(broker):
    times = list()
    times.append(time())
    node_subgraph = set()
    edge_subgraph = dict()

    node_subgraph.add(broker)
    edge_subgraph[broker] = set()
    for adj in ADJ[broker]:
        edge_subgraph[broker].add(adj)
        edge_subgraph[adj] = set()
        node_subgraph.add(adj)
        for adj2 in ADJ[adj]:
            edge_subgraph[adj].add(adj2)
            node_subgraph.add(adj2)
            if adj2 not in edge_subgraph.keys():
                edge_subgraph[adj2] = set()

    hop2_nodes = node_subgraph - set(ADJ[broker]) - {broker}

    for node in hop2_nodes:
        for adj in ADJ[node]:
            if adj in node_subgraph:
                edge_subgraph[node].add(adj)

    node_count = len(node_subgraph)
    edge_count = 0
    for node in node_subgraph:
        edge_count += len(edge_subgraph[node])

    #print('subgraph_size : {} nodes, {} edges'.format(node_count, edge_count))

    times.append(time())
    #print_time('get subgraph')

    unfinished = copy.deepcopy(node_subgraph)
    #random.shuffle(unfinished)
    #ttl = 3000000000
    ttl = 500000000
    bfs_count = 0

    bc = dict()
    for node in node_subgraph:
        bc[node] = 0
    while ttl > 0 and len(unfinished) > 0:
        bfs_count += 1
        if bfs_count % 10000 == 0:
            print('BFS count : {}'.format(bfs_count))
        V = unfinished.pop()
        ttl -= edge_count
        Q = Queue()
        Q.put(V)
        S = list()
        P = dict()
        sigma = dict()
        dst = dict()
        delta = dict()
        for node in node_subgraph:
            P[node] = list()
            sigma[node] = 0
        sigma[V] = 1
        dst[V] = 0

        while not Q.empty():
            v = Q.get()
            S.append(v)
            for n in edge_subgraph[v]:
                if n not in dst:
                    Q.put(n)
                    dst[n] = dst[v] + 1
                else:
                    pass
                if dst[n] == dst[v] + 1:
                    sigma[n] += sigma[v]
                    P[n].append(v)

        for n in sigma.keys():
            if sigma[n] == 0:
                continue
            else:
                delta[n] = 1 / sigma[n]

        while S:
            w = S.pop()
            for n in P[w]:
                delta[n] += delta[w]
            if w != V:
                bc[w] += delta[w] * sigma[w] - 1

    #print('finished BFS : {} / {}'.format(len(node_subgraph) - len(unfinished), len(node_subgraph)))

    return bc, unfinished

def decide_landmarks(broker):
    #print('Broker : ', broker)

    landmarks = list()
    bc, unfinished = calc_bc(broker)
    bc_sorted = sorted(bc.items(), key=lambda x:x[1], reverse=True)

    sum = 0
    for node in bc_sorted[:10]:
        sum += node[1]
        #print(node[0], node[1])
    tmp = 0
    for node in bc_sorted[:10]:
        landmarks.append(node[0])
        tmp += node[1]
        if tmp > sum * 0.5:
            break

    #print('Landmarks : ', landmarks)
    #return landmarks, bc, unfinished
    return landmarks, bc_sorted, unfinished

def BFS(source):
    Q = Queue()
    Q.put(source)
    S = list()
    P = {source : list()}
    sigma = {source: 1}
    dst = {source: 0}
    score = {source: 0}
    delta = dict()

    while not Q.empty():
        v = Q.get()
        S.append(v)
        for node in ADJ[v]:
            if node not in dst:
                Q.put(node)
                dst[node] = dst[v] + 1
                sigma[node] = 0
                P[node] = list()
                score[node] = 0
            else:
                pass
            if dst[node] == dst[v] + 1:
                sigma[node] += sigma[v]
                P[node].append(v)

    for node in sigma.keys():
        delta[node] = 1 / sigma[node]

    while S:
        w = S.pop()
        for node in P[w]:
            delta[node] += delta[w]
        if w != source:
            score[w] += delta[w] * sigma[w] - 1

    return score, dst

def BFS2(source, area_count):
    adj_sub = get_subgraph(source, area_count)

    Q = Queue()
    Q.put(source)
    S = list()
    P = {source : list()}
    sigma = {source: 1}
    dst = {source: 0}
    score = {source: 0}
    delta = dict()

    while not Q.empty():
        v = Q.get()
        S.append(v)
        for node in adj_sub[v]:
            if node not in dst:
                Q.put(node)
                dst[node] = dst[v] + 1
                sigma[node] = 0
                P[node] = list()
                score[node] = 0
            else:
                pass
            if dst[node] == dst[v] + 1:
                sigma[node] += sigma[v]
                P[node].append(v)

    for node in sigma.keys():
        delta[node] = 1 / sigma[node]

    while S:
        w = S.pop()
        for node in P[w]:
            delta[node] += delta[w]
        if w != source:
            score[w] += delta[w] * sigma[w] - 1

    return score, dst

def decide_exact_brokers(source, dst_of_broker):
    score, dst = BFS(source)
    extracted = dict()
    for node in score.keys():
        if dst[node] == dst_of_broker:
            extracted[node] = score[node]
    score_sorted = sorted(extracted.items(), key=lambda x:x[1], reverse=True)

    return score_sorted

def decide_exact_brokers2(source, dst_of_broker, area_count):
    score, dst = BFS2(source, area_count)
    extracted = dict()
    for node in score.keys():
        if dst[node] == dst_of_broker:
            extracted[node] = score[node]
    score_sorted = sorted(extracted.items(), key=lambda x:x[1], reverse=True)

    return score_sorted

def remove_loop(path):
    no_loop_path = list()
    for node in path:
        if node in no_loop_path:
            index = no_loop_path.index(node)
            del no_loop_path[(index+1):]
        else:
            no_loop_path.append(node)

    return no_loop_path

def decide_random_brokers(source, dst_of_broker, node_count):
    def add_adj(v1, v2):
        if v1 not in detected_edge:
            detected_edge[v1] = set()
        if v2 not in detected_edge:
            detected_edge[v2] = set()
        detected_edge[v1].add(v2)
        detected_edge[v2].add(v1)

    score = dict()
    for node in ADJ.keys():
        score[node] = 0

    paths = []
    visited = {source}
    max_step = 10000
    dst = {source : 0}
    Q = Queue()
    detected_edge = {source : set()}
    for i in range(node_count):
        walker = source
        paths.append([source])
        for j in range(1, max_step):
            previous = walker
            walker = random.choice(ADJ[walker])
            paths[i].append(walker)

            if walker in detected_edge[previous]:
                continue
            else:
                add_adj(previous, walker)

            if walker not in dst.keys():
                dst[walker] = dst[previous] + 1
            elif dst[walker] < dst[previous] - 1:
                dst[previous] = dst[walker] + 1
                Q.put(previous)
            elif dst[walker] > dst[previous] + 1:
                dst[walker] = dst[previous] + 1
                Q.put(walker)
            else:
                pass

            while not Q.empty():
                v = Q.get()
                for node in detected_edge[v]:
                    if dst[node] > dst[v] + 1:
                        dst[node] = dst[v] + 1
                        Q.put(node)

            if walker not in visited:
                visited.add(walker)
                break

    for path in paths:
        path = remove_loop(path)
        for node in path:
            score[node] += 1

    #print(paths)

    extracted = dict()
    for node in dst.keys():
        if dst[node] == dst_of_broker:
            extracted[node] = score[node]
    score_sorted = sorted(extracted.items(), key=lambda x:x[1], reverse=True)

    return score_sorted

def get_visited_nodes(source, dst_of_broker, node_count):
    def add_adj(v1, v2):
        if v1 not in detected_edge:
            detected_edge[v1] = set()
        if v2 not in detected_edge:
            detected_edge[v2] = set()
        detected_edge[v1].add(v2)
        detected_edge[v2].add(v1)

    score = dict()
    for node in ADJ.keys():
        score[node] = 0

    paths = []
    visited = {source}
    max_step = 10000
    dst = {source : 0}
    Q = Queue()
    detected_edge = {source : set()}
    for i in range(node_count):
        walker = source
        paths.append([source])
        for j in range(1, max_step):
            previous = walker
            walker = random.choice(ADJ[walker])
            paths[i].append(walker)

            if walker in detected_edge[previous]:
                continue
            else:
                add_adj(previous, walker)

            if walker not in dst.keys():
                dst[walker] = dst[previous] + 1
            elif dst[walker] < dst[previous] - 1:
                dst[previous] = dst[walker] + 1
                Q.put(previous)
            elif dst[walker] > dst[previous] + 1:
                dst[walker] = dst[previous] + 1
                Q.put(walker)
            else:
                pass

            while not Q.empty():
                v = Q.get()
                for node in detected_edge[v]:
                    if dst[node] > dst[v] + 1:
                        dst[node] = dst[v] + 1
                        Q.put(node)

            if walker not in visited:
                visited.add(walker)
                break

    for path in paths:
        path = remove_loop(path)
        for node in path:
            score[node] += 1

    #print(paths)

    extracted = dict()
    for node in dst.keys():
        if dst[node] == dst_of_broker:
            extracted[node] = score[node]
    score_sorted = sorted(extracted.items(), key=lambda x:x[1], reverse=True)

    return score_sorted, visited
