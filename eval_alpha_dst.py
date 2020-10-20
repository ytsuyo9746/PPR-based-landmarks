from functions import *
from Graph import *
import csv

directory = './dataset/web-Google/'
graph_flp, graph_rw = create_graph_object(directory)

alpha_for_bro = 0.15
epsilon_for_bro = 0.0001
alpha_for_land=0.15
epsilon_for_land=0.0001
bro_count=5
land_count_per_bro=1
subgraph_size=2
bro_to_lands = dict()
f = open(directory + 'broker_to_landmarks.json')
bro_to_lands = json.load(f)
src_to_bros = dict()
# f = open(directory + 'source_to_brokers.json')
# src_to_bros = json.load(f)
eval_count_per_random_count = 5

source_ids = ['301955', '841731', '193741', '541390', '23433', '702146',\
'595098', '329856', '640242', '840775', '264703', '651994', '21799', '437238']

random_count = [100, 300, 1000, 3100, 10000, 18000, \
30000, 60000, 100000, 180000, 320000, 560000]

# alphas = [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.015]

bro_dst = dict()
land_dst = dict()
for node_id in source_ids:
    bro_dst[node_id] = dict()
    land_dst[node_id] = dict()
    for alpha in alphas:
        bro_dst[node_id][alpha] = 0
        land_dst[node_id][alpha] = 0

for source_id in source_ids:
    print('source: {}'.format(source_id))
    source_node = graph_flp.search_node(source_id)
    for alpha in alphas:
        print('alpha: {}'.format(alpha))
        exact_P, exact_R = calc_PPR_by_forward_local_push(graph_flp, source_id, alpha, epsilon_for_bro)
        exact_P_sorted = sorted(exact_P.items(), key=lambda x:x[1], reverse=True)
        exact_brokers = list()
        for node_id, score in exact_P_sorted[:bro_count]:
            exact_brokers.append(node_id)
        exact_landmarks = list()
        for node_id in exact_brokers:
            exact_landmarks += decide_landmarks_from_broker(graph_flp, node_id, alpha, epsilon_for_land,\
            land_count_per_bro, subgraph_size, bro_to_lands)

        for broker_id in exact_brokers:
            broker_node = graph_flp.search_node(broker_id)
            bro_dst[source_id][alpha] += graph_flp.get_dst_between(source_node, broker_node) / bro_count
        for landmark_id in exact_landmarks:
            landmark_node = graph_flp.search_node(landmark_id)
            land_dst[source_id][alpha] += graph_flp.get_dst_between(source_node, landmark_node) / (bro_count * land_count_per_bro)
        print("broker: {}, landmark: {}".format(bro_dst[source_id][alpha], land_dst[source_id][alpha]))

timestamp = get_timestamp()

with open(directory + 'data/' + 'eval_alpha_dst_broker' + timestamp + '.json', 'w') as fp:
    json.dump(bro_dst, fp)

with open(directory + 'data/' + 'eval_alpha_dst_landmark' + timestamp + '.json', 'w') as fp:
    json.dump(land_dst, fp)

with open(directory + 'broker_to_landmarks.json', 'w') as fp:
    json.dump(bro_to_lands, fp)

# with open(directory + 'source_to_brokers.json', 'w') as fp:
#     json.dump(src_to_bros, fp)

file = open(directory + 'data/' + 'eval_alpha_dst' + timestamp + '.csv', 'w')
w = csv.writer(file)
w.writerow(['alpha'] + alphas)
for source_id in source_ids:
    bro_list = [source_id]
    land_list = [source_id]
    for alpha in alphas:
        bro_list.append(bro_dst[source_id][alpha])
        land_list.append(land_dst[source_id][alpha])
    w.writerow(bro_list)
    w.writerow(land_list)
file.close()
