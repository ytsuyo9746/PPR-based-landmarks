from functions import *
from Graph import *

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

rate = dict()
for node_id in source_ids:
    rate[node_id] = dict()
    for count in random_count:
        rate[node_id][count] = 0

for source_id in source_ids:
    print('source: {}'.format(source_id))
    exact_landmarks = decide_exact_landmark(graph_flp, source_id, alpha_for_bro, epsilon_for_bro, \
    alpha_for_land, epsilon_for_land, bro_count, land_count_per_bro, subgraph_size, bro_to_lands)
    for count in random_count:
        print('count: {}'.format(count))
        for i in range(eval_count_per_random_count):
            random_landmarks = decide_random_landmark(graph_rw, source_id, alpha_for_bro, count, \
            alpha_for_land, epsilon_for_land, bro_count, land_count_per_bro, subgraph_size, bro_to_lands)
            rate[source_id][count] += match_rate(exact_landmarks, random_landmarks) / eval_count_per_random_count


with open(directory + 'broker_to_landmarks.json', 'w') as fp:
    json.dump(bro_to_lands, fp)
