import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def prob_hop_len(alpha, hop_len):
    return ((1 - alpha)**(hop_len - 1)) * alpha

def calc_exp(alpha):
    exp = 0
    for i in range(1, 1000):
        exp += i * prob_hop_len(alpha, i)
    return exp

alphas = np.arange(0.01, 1.0, 0.01)
exps = list()

for alpha in alphas:
    exp = calc_exp(alpha)
    exps.append(exp)
    # print('Expectation of hop length for alpha = {}: {}'.format(alpha, exp))

plt.plot(alphas, exps)
plt.xlim(0,1)
plt.ylim(0,100)
plt.xlabel('起点回帰確率 α')
plt.ylabel('平均経路長')
# plt.xscale('log')
# plt.yscale('log')
plt.show()
