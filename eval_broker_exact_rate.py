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
# f = open(directory + 'broker2landmarks.json')
# bro_to_lands = json.load(f)
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
    exact_P, exact_R = calc_PPR_by_forward_local_push(graph_flp, source_id, alpha_for_bro, epsilon_for_bro)
    exact_P_sorted = sorted(exact_P.items(), key=lambda x:x[1], reverse=True)
    exact_brokers = list()
    for node_id, score in exact_P_sorted[:bro_count]:
        exact_brokers.append(node_id)

    for count in random_count:
        print('count: {}'.format(count))
        for i in range(eval_count_per_random_count):
            random_P = calc_PPR_by_random_walk(graph_rw, source_id, alpha_for_bro, count)
            random_P_sorted = sorted(random_P.items(), key=lambda x:x[1], reverse=True)
            random_brokers = list()
            for node_id, score in random_P_sorted[:bro_count]:
                random_brokers.append(node_id)
            rate[source_id][count] += match_rate(exact_brokers, random_brokers) / eval_count_per_random_count


with open(directory + 'data/' + 'eval_broker_exact_rate' + get_timestamp() + '.json', 'w') as fp:
    json.dump(rate, fp)
