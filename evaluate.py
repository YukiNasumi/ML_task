import numpy as np
import argparse
target = np.genfromtxt('data/targets.csv')
base_list = [1, 5, 10, 100]
parser = argparse.ArgumentParser()
parser.add_argument('--t',type=str,required=False,default='experiments/')
args = parser.parse_args()
target_dir = args.t
if not target_dir[-1]=='/':
    target_dir+='/'
for base_num in base_list:
    acc = []
    for i in range(1, 11):
        fold = np.genfromtxt(target_dir+'base%d_fold%d.csv' % (base_num, i), delimiter=',', dtype=int)
        accuracy = sum(target[fold[:, 0] - 1] == fold[:, 1]) / fold.shape[0]
        acc.append(accuracy)

    print(np.array(acc).mean())
