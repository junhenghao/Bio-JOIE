from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), './src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
from numpy import linalg as LA
import tensorflow as tf
import time
import multiG  
import model2 as model
from tester1 import Tester
import argparse

# all parameter required
parser = argparse.ArgumentParser(description='JOIE Testing: Type Linking')
parser.add_argument('--modelname', type=str,help='model category')
parser.add_argument('--model', type=str,help='model name including data and model')
parser.add_argument('--testfile', type=str,help='test data')
parser.add_argument('--method', type=str,help='embedding method used')
parser.add_argument('--resultfolder', type=str,help='result output folder')
parser.add_argument('--graph', type=str,help='test which graph (ins/onto)')
parser.add_argument('--GPU', type=str, default='0', help='GPU ID')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

path_prefix = './model/'+args.modelname
hparams_str = args.model
args.method, args.bridge =  hparams_str.split('_')[0], hparams_str.split('_')[1]
model_file = path_prefix+"/"+hparams_str+'/'+'multiKG-model2.ckpt'
data_file = path_prefix+"/"+hparams_str+'/'+'multiKG-data.bin'
test_data = args.testfile
result_folder = './'+args.resultfolder+'/'+args.modelname
result_file = result_folder+"/"+hparams_str+"_graph_"+args.graph+"_result.txt"

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

topK = 10

tester = Tester()
tester.build(save_path = model_file, data_save_path = data_file, graph=args.graph, method=args.method, bridge=args.bridge)
tester.load_test_link(test_data, max_num=15000, splitter = '\t', line_end = '\n')

import multiprocessing
from multiprocessing import Process, Value, Lock, Manager

cpu_count = multiprocessing.cpu_count()

manager = Manager()

index = Value('i', 0, lock=True) #index
rst_predict = manager.list() #scores for each case
t0 = time.time()

def test(tester, index, rst_predict):
    while index.value < len(tester.test_triples): # and index.value < 15000:
        idx = index.value
        index.value += 1
        if idx > 0 and idx % 1000 == 0:
            print("Tested %d in %d seconds." % (idx+1, time.time()-t0))
            try:
                print(np.mean(rst_predict, axis=0))
            except:
                pass
        e1, r, e2 = tester.test_triples[idx]
        rst = tester.kNN_rels(e1, e2)

        this_hit = []
        hit = 0.
        strl = tester.rel_index2str(rst[0][0], tester.graph_id)
        strr = tester.rel_index2str(r, tester.graph_id)
        this_index = 0
        this_rank = None
        for pr in rst:
            this_index += 1
            if (hit < 1. and (pr[0] == r or pr[0] in tester.test_ht_map[e1][e2])) or (hit < 1. and tester.rel_index2str(pr[0], tester.graph_id) == strr):
                hit = 1.
                this_rank = this_index
            this_hit.append(hit)
        hit_first = 0
        if rst[0][0] == r or rst[0][0] in tester.test_ht_map[e1][e2] or strl == strr:
            hit_first = 1
        rst_predict.append(np.array(this_hit))


# tester.rel_num_cases
processes = [Process(target=test, args=(tester, index, rst_predict)) for x in range(2 - 1)]
for p in processes:
    p.start()
for p in processes:
    p.join()

hits = np.mean(rst_predict, axis=0)

# print out result file
fp = open(result_file, 'w')
fp.write("Hits@"+str(topK)+'\n')
print(hits)
fp.write(' '.join([str(x) for x in hits]) + '\n')
fp.close()