
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os

import numpy as np
import tensorflow as tf
import argparse

# add path
sys.path.append(os.path.join(os.path.dirname(__file__), './src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from KG import KG
from multiG import multiG   # we don't import individual things in a model. This is to make auto reloading in Notebook happy
import model2 as model
from trainer2 import Trainer

# make param str function including all hyper parameters
def make_hparam_string(method, bridge,dim1, dim2, a1, a2, m1, mA, mT, fold_a, fold_t, weight, view):
	# input params: dim, onto_ratio, type_ratio, lr, 
	return "%s_%s_dim1_%s_dim2_%s_a1_%s_a2_%s_m1_%s_mA_%s_mT_%s_%s_fold_%s.%s_view_%s" % (method, bridge, dim1, dim2, a1, a2, m1, mA, mT, weight, fold_a, fold_t, view)

# parameter parsing
parser = argparse.ArgumentParser(description='JOIE Training')
# required parameters
parser.add_argument('--method', type=str, help='embedding method(transe/distmult/hole)')
parser.add_argument('--bridge', type=str, help='entity-conept link method(CG/CMP-linear/CMP-single)')
parser.add_argument('--kg1f', type=str, help='KG1 file path (triple file)')
parser.add_argument('--kg2f', type=str, help='KG2 file path (triple file)')
parser.add_argument('--alignf', type=str, help='type link file path (triple file)') # triple with relation "type"
parser.add_argument('--taxf', type=str, help='taxonomy file path (triple file)') 
parser.add_argument('--modelname', type=str,help='model name and data path')
# default hyper-parameters
parser.add_argument('--epochs', type=int, default=100,help='Number of epochs')
parser.add_argument('--GPU', type=str, default='0', help='GPU ID')
parser.add_argument('--dim1', type=int, default=300,help='Dimension for KG1')
parser.add_argument('--dim2', type=int, default=100,help='Dimension for KG2')
parser.add_argument('--batch_K1', type=int, default=256,help='Batch size for KG1')
parser.add_argument('--batch_K2', type=int, default=64,help='Batch size for KG2')
parser.add_argument('--batch_A', type=int, default=128,help='Batch size for alignf')
parser.add_argument('--batch_T', type=int, default=64,help='Batch size for taxf')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--a1', type=float, default=0.5, help='lr ratio 1 AM/KM')
parser.add_argument('--a2', type=float, default=0.5, help='lr ratio 2 TM/KM') # probably not used
parser.add_argument('--m1', type=float, default=1.0, help='margin param KG1')
parser.add_argument('--m2', type=float, default=1.0, help='margin param KG2')
parser.add_argument('--mA', type=float, default=0.5, help='margin param Align')
parser.add_argument('--mT', type=float, default=0.5, help='margin param Tax')
parser.add_argument('--L1', type=bool, default=False, help='whether to use L1 norm')
parser.add_argument('--weight', type=str, default=None, help='GoTerm alignment weight option')
parser.add_argument('--AMfold', type=int, default=2, help='number of AM update per KG')
parser.add_argument('--TMfold', type=int, default=2, help='number of TM update per KG')
parser.add_argument('--view', type=str, default='all', help='GPU ID')
args = parser.parse_args()

# check parameters
if args.bridge == "CG" and args.dim1 != args.dim2: #CG does not allow different dims
	print("Warning! CG does not allow ")
if args.method not in ['transe','distmult','hole']:
	raise ValueError("Embedding method not valid!")
if args.bridge not in ['CG','CMP-linear','CMP-single','CMP-double']:
	raise ValueError("Bridge method not valid!")

print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU #specify GPU id

modelname = args.modelname
path_prefix = './model/'+modelname+'/'
hparams_str = make_hparam_string(args.method, args.bridge, args.dim1, args.dim2, args.a1, args.a2, args.m1, args.mA, args.mT, args.weight, args.AMfold, args.TMfold, args.view)
model_prefix = path_prefix+hparams_str

model_path = model_prefix+'/'+'multiKG-model2.ckpt'
data_path = model_prefix+'/'+'multiKG-data.bin'

if not os.path.exists(model_prefix):
    os.makedirs(model_prefix)
else:
	raise ValueError("Overwrite Warning: model directory has already existed!")

# intialize graphs
KG1 = KG()
KG2 = KG()
KG1.load_triples(filename = args.kg1f, splitter = '\t', line_end = '\n', if_onto_graph=False)
KG2.load_triples(filename = args.kg2f, splitter = '\t', line_end = '\n', if_onto_graph=True)
this_data = multiG(KG1, KG2)
this_data.load_taxonomy(filename = args.taxf, splitter='\t', line_end='\n')
this_data.load_align(filename = args.alignf, lan1 = 'instance', lan2 = 'ontology', splitter = '\t', line_end = '\n')

# start trainer
m_train = Trainer()
#udpate dim
m_train.build(this_data, method=args.method, bridge=args.bridge, dim1=args.dim1, dim2=args.dim2, 
	batch_sizeK1=args.batch_K1, batch_sizeK2=args.batch_K2, batch_sizeA=args.batch_A, batch_sizeT=args.batch_T,
	a1=args.a1, a2=args.a2, m1=args.m1, m2=args.m2, mA=args.mA, mT=args.mT, 
	model_save_path = model_path, data_save_path = data_path, L1=args.L1, w_opt=args.weight)
m_train.train(epochs=args.epochs, save_every_epoch=10, lr=args.lr, AM_fold=args.AMfold, TM_fold=args.TMfold)




