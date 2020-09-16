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
eval_count_per_random_count = 5

source_ids = ['301955', '841731', '193741', '541390', '23433', '702146',\
'595098', '329856', '640242', '840775', '264703', '651994', '21799', '437238']

random_count = [100, 300, 1000, 3100, 10000, 18000, \
30000, 60000, 100000, 180000, 320000, 560000]

bro_rate = dict()
land_rate = dict()
for node_id in source_ids:
    bro_rate[node_id] = dict()
    land_rate[node_id] = dict()
    for count in random_count:
        bro_rate[node_id][count] = 0
        land_rate[node_id][count] = 0

for source_id in source_ids:
    print('source: {}'.format(source_id))
    exact_P, exact_R = calc_PPR_by_forward_local_push(graph_flp, source_id, alpha_for_bro, epsilon_for_bro)
    exact_P_sorted = sorted(exact_P.items(), key=lambda x:x[1], reverse=True)
    exact_brokers = list()
    exact_landmarks = list()
    for node_id, score in exact_P_sorted[:bro_count]:
        exact_brokers.append(node_id)
        exact_landmarks += decide_landmarks_from_broker(graph_flp, node_id, alpha_for_land, epsilon_for_land,\
        land_count_per_bro, subgraph_size, bro_to_lands)

    for count in random_count:
        print('count: {}'.format(count))
        for i in range(eval_count_per_random_count):
            random_P = calc_PPR_by_random_walk(graph_rw, source_id, alpha_for_bro, count)
            random_P_sorted = sorted(random_P.items(), key=lambda x:x[1], reverse=True)
            random_brokers = list()
            random_landmarks = list()
            for node_id, score in random_P_sorted[:bro_count]:
                random_brokers.append(node_id)
                random_landmarks += decide_landmarks_from_broker(graph_flp, node_id, alpha_for_land, epsilon_for_land,\
                land_count_per_bro, subgraph_size, bro_to_lands)

            bro_rate[source_id][count] += match_rate(exact_brokers, random_brokers) / eval_count_per_random_count
            land_rate[source_id][count] += match_rate(exact_landmarks, random_landmarks) / eval_count_per_random_count

timestamp = get_timestamp()

with open(directory + 'data/' + 'eval_broker_exact_rate' + timestamp + '.json', 'w') as fp:
    json.dump(bro_rate, fp)

with open(directory + 'data/' + 'eval_landmark_exact_rate' + timestamp + '.json', 'w') as fp:
    json.dump(land_rate, fp)

file = open(directory + 'data/' + 'eval_random_rate' + timestamp + '.csv', 'w')
w = csv.writer(file)
for source_id in source_ids:
    bro_list = list()
    land_list = list()
    for count in random_count:
        bro_list.append(bro_rate[source_id][count])
        land_list.append(land_rate[source_id][count])
    w.writerow(bro_list)
    w.writerow(land_list)
file.close()
