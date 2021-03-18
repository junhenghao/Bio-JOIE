# some useful tools
import tensorflow as tf
import numpy as np
from numpy.fft import fft, ifft
import numpy as np
import networkx as nx
import random
random.seed(525)

def circular_correlation(h, t):
    return tf.real(tf.spectral.ifft(tf.multiply(tf.conj(tf.spectral.fft(tf.complex(h, 0.))), tf.spectral.fft(tf.complex(t, 0.)))))

def np_ccorr(h, t):
    return ifft(np.conj(fft(h)) * fft(t)).real

# for goterm
class GO_Tree(object):
    # create 
    def __init__(self, filepath):
        self.full_tree_path = filepath
        self.GoTree = None
        self.roots = ["GO:0008150","GO:0005575","GO:0003674"] # pre-defined
        self.load_graph(filepath)
        
    # construct the parent-children tree only by "is_a" relation
    def load_graph(self, filepath, auto_root=False):
        self.GoTree = nx.DiGraph()
        print("loading tree edges from ", self.full_tree_path)
        for line in open(filepath):
            line = line.strip().replace('\r','').replace('_',':').split('\t')
            if len(line) != 3:
                continue
            cid , pid = line[0], line[2]
            if not self.GoTree.has_node(cid):
                self.GoTree.add_node(cid)
            if not self.GoTree.has_node(pid):
                self.GoTree.add_node(pid)
            # add both directions
            self.GoTree.add_edge(cid, pid, relname="has/to Parent")
            # self.GoTree.add_edge(pid, cid, relname="has/to Child")
        print("GoTerm Tree constructed: [#nodes] {0} [#edges] {1}"
              .format(self.GoTree.number_of_nodes(), self.GoTree.number_of_edges()))
        if auto_root:
            self.roots =  [n for n,d in self.GoTree.out_degree() if d==0] 
        print("roots:",self.roots)
    
    def find_root_node(self, qid):
        qid = qid.replace('_', ':')
        try:
            qid_parents = nx.bfs_tree(self.GoTree, qid, reverse=False)
            qid_roots = set(self.roots).intersection(set(qid_parents))
            if len(list(qid_roots)) == 0:
                qid_roots_str = 'unknown'
            elif len(list(qid_roots)) > 1:
                qid_roots_str  = 'multiple'
            else:
                qid_roots_str  = list(qid_roots)[0].replace(':', '_')
        except:
            qid_roots_str = 'unknown'
        return qid_roots_str
    
    def find_all_direct_parents(self, qid):
        parent_tree = nx.bfs_tree(self.GoTree, qid, reverse=False)
        return set(parent_tree.nodes) - set([qid])
            
    def find_all_direct_children(self, qid):
        children_tree = nx.bfs_tree(self.GoTree, qid, reverse=True)
        return set(children_tree.nodes) - set([qid])
    
    def find_siblings_set(self, qid, parent_level=2, children_level=2, verbose=False): 
        parent_tree = nx.bfs_tree(self.GoTree, qid, reverse=False, depth_limit=parent_level)
        selected_parent = set(parent_tree.nodes) - set([qid])
        sibling_set = set([])
        for item in selected_parent:
            subtree = nx.bfs_tree(self.GoTree, item, reverse=True, depth_limit=children_level)
            cur_siblings = set(subtree.nodes) - set([item])
            sibling_set = sibling_set.union(cur_siblings)
        if verbose:
            print("Parent level up:{0}, total siblings: {1}".format(len(selected_parent), len(sibling_set)))
        return sibling_set
    
    def generate_regular_negative(self, input_pairs, maxnum=10, ratio=1.0, rand_seed=None):
        if ratio > maxnum:
            raise ValueError("max < ratio not allowed!")
        np.random.seed(seed=rand_seed)
        hardneg_pairs = []
        for item in input_pairs:
            if len(item) != 2:
                continue
            pro, cid = item[0], item[1]
            if cid not in list(self.GoTree.nodes()):
                continue
            pset = self.find_all_direct_parents(cid)
            cset = self.find_all_direct_children(cid)
            negset = set(self.GoTree.nodes) - pset - cset
            # print("find negnet", len(negset))
            # number of neg control
            if len(negset) > maxnum: 
                negset = random.sample(negset, maxnum)
            for k in negset:
                hardneg_pairs.append((pro, k))
        hardneg_pairs = np.array(hardneg_pairs)
        print(hardneg_pairs.shape)
        # ratio selection
        if len(hardneg_pairs) < len(input_pairs) * ratio:
            print("negative samples less than pos * ratio! Neg. Pairs #", len(hardneg_pairs))
            return hardneg_pairs
        else:
            hardneg_pairs_rand = hardneg_pairs[np.random.choice
                                               (len(hardneg_pairs), int(len(input_pairs)*ratio), replace=False)]
            print("Neg. Pairs #", len(hardneg_pairs))
            return hardneg_pairs_rand
            
    
    def generate_hard_negative(self, input_pairs, maxnum=10, ratio=1.0, rand_seed=None):
        if ratio > maxnum:
            raise ValueError("max < ratio not allowed!")
        np.random.seed(seed=rand_seed)
        hardneg_pairs = []
        for item in input_pairs:
            if len(item) != 2:
                continue
            pro, cid = item[0], item[1]
            if cid not in list(self.GoTree.nodes()):
                continue
            sibset = self.find_siblings_set(cid, parent_level=2, children_level=2)
            if len(sibset) > maxnum:
                sibset = random.sample(sibset, maxnum)
            print("find sibnet", len(sibset))
            for sib in sibset:
                hardneg_pairs.append((pro, sib))
        hardneg_pairs = np.array(hardneg_pairs)  
        # ratio selection
        if len(hardneg_pairs) < len(input_pairs) * ratio:
            print("negative samples less than pos * ratio! Neg. Pairs #", len(hardneg_pairs))
            return hardneg_pairs
        else:
            hardneg_pairs_rand = hardneg_pairs[np.random.choice
                                               (len(hardneg_pairs), int(len(input_pairs)*ratio), replace=False)]
            print("Neg. Pairs #", len(hardneg_pairs))
            return hardneg_pairs_rand