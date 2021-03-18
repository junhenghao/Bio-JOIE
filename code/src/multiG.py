"""Processing of data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pickle
import time
from KG import KG
import networkx as nx
import Queue
#from multiprocessing import Queue
import time
from tqdm import tqdm

# ================= update function on graph ===================
def updateUp_height(graph, startNode, height_dict):
    nodeQueue = Queue.Queue()
    #nodeQueue = Queue()
    nodeQueue.put(startNode)
    while not nodeQueue.empty():
        curNode = nodeQueue.get()
        curChildList = [v[1] for v in  list(nx.bfs_edges(graph, source=curNode, reverse=False, depth_limit=1))]
        curChildHeight = [height_dict[v] for v in curChildList]
        curNodeHeight = height_dict[curNode]
        height_dict[curNode] = max(curChildHeight) + 1
        if height_dict[curNode] != curNodeHeight: # put all current parents into queue for update
            curParentList = [v[1] for v in  list(nx.bfs_edges(graph, source=curNode, reverse=True, depth_limit=1))]
            for item in curParentList:
                nodeQueue.put(item)
                #time.sleep(0.1)
    return height_dict

def updateDown_depth(graph, startNode, depth_dict):
    nodeQueue = Queue.Queue()
    #nodeQueue = Queue()
    nodeQueue.put(startNode)
    while not nodeQueue.empty():
        curNode = nodeQueue.get()
        curParentList = [v[1] for v in  list(nx.bfs_edges(graph, source=curNode, reverse=True, depth_limit=1))]
        curParentDepth = [depth_dict[v] for v in curParentList]
        curNodeDepth = depth_dict[curNode]
        depth_dict[curNode] = max(curParentDepth) + 1
        if depth_dict[curNode] != curNodeDepth: # put all current parents into queue for update
            curChildList = [v[1] for v in  list(nx.bfs_edges(graph, source=curNode, reverse=False, depth_limit=1))]
            for item in curChildList:
                nodeQueue.put(item) 
                #time.sleep(0.1)
    return depth_dict
# ==============================================================

class multiG(object):
    def __init__(self, KG1=None, KG2=None):
        if KG1 == None or KG2 == None:
            self.KG1 = KG()
            self.KG2 = KG()
        else:
            self.KG1 = KG1
            self.KG2 = KG2
        self.lan1 = 'en'
        self.lan2 = 'fr'
        self.align = np.array([0])
        self.align_desc = np.array([0])
        self.aligned_KG1 = set([])
        self.aligned_KG2 = set([])
        self.aligned_KG1_index = np.array([0])
        self.aligned_KG2_index = np.array([0])
        self.unaligned_KG1_index = np.array([0])
        self.unaligned_KG2_index = np.array([0])
        self.align_valid = np.array([0])
        self.n_align = 0
        self.n_align_desc = 0
        self.ent12 = {}
        self.ent21 = {}
        # ===========================
        # add taxonomy store
        self.taxonomy = np.array([0])
        self.con12 = {}
        self.con21 = {}
        self.n_taxonomy = 0
        # ===========================
        self.batch_sizeK1 = 1024
        self.batch_sizeK2 = 64
        self.batch_sizeA = 32
        self.L1 = False
        self.dim1 = 300 #stored for TF_Part
        self.dim2 = 100
        self.GoTree = nx.DiGraph() # completed after loading taxonomy and alignment
        self.max_proDegree = 1 # initialized as -1 (updated after loading alignment)
        self.min_proDegree = 1 # initialized as -1 (updated after loading alignment)

    def load_align(self, filename, lan1 = 'en', lan2 = 'fr', splitter = '@@@', line_end = '\n', desc=False):
        '''Load the dataset.'''
        print("current size: [ entity: " + str(len(self.KG1.ents)) +"] [ concept: "+str(len(self.KG2.ents))+"]")
        weight = 1.
        align = []
        last_c = -1
        last_r = -1
        self.n_align = 0
        self.n_align_desc = 0
        self.align = []
        if desc:
            self.align_desc = []
        temp_proDegree = nx.get_node_attributes(self.GoTree, name='connect')
        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            try:
                e1 = self.KG1.ent_str2index(line[0])
                e2 = self.KG2.ent_str2index(line[2]) # pairs in GOTerm and protein (triples again)
                assert e1 != None and e2 != None
                # ========================================================
                # original version 
                # TODO: Should adopt new concepts which only in taxonomy
                # ========================================================
                '''
                if e1 == None:
                    current_entnum = len(self.KG1.ents)
                    self.KG1.ents[current_entnum] = line[0]
                    self.KG1.index_ents[line[0]] = current_entnum
                    self.KG1.ent_tokens[current_entnum] = set(line[0].replace('(','').replace(')','').split(' '))
                    e1 = self.KG1.ent_str2index(line[0])
                if e2 == None:
                    current_entnum = len(self.KG2.ents)
                    self.KG2.ents[current_entnum] = line[1]
                    self.KG2.index_ents[line[1]] = current_entnum
                    self.KG2.ent_tokens[current_entnum] = set(line[1].replace('(','').replace(')','').split(' '))
                    e2 = self.KG2.ent_str2index(line[1])
                '''
                # === Load node properties ====
                assert(self.GoTree.has_node(e2))
                if self.GoTree.has_node(e2):
                    temp_proDegree[e2] += 1
                # === End: load properties ====

                self.align.append((e1, e2))
                self.aligned_KG1.add(e1)
                self.aligned_KG2.add(e2)
                if self.ent12.get(e1) == None:
                    self.ent12[e1] = set([e2])
                else:
                    self.ent12[e1].add(e2)
                if self.ent21.get(e2) == None:
                    self.ent21[e2] = set([e1])
                else:
                    self.ent21[e2].add(e1)
                self.n_align += 1
                if desc:
                    if (not self.KG1.get_desc_embed(e1) is None) and (not self.KG2.get_desc_embed(e2) is None):
                        self.align_desc.append((e1, e2))
                        self.n_align_desc += 1
            except:
                continue
        self.align = np.array(self.align)
        if desc:
            self.align_desc = np.array(self.align_desc)
        self.aligned_KG1_index = np.array([e for e in self.aligned_KG1])
        self.aligned_KG2_index = np.array([e for e in self.aligned_KG2])
        self.unaligned_KG1_index, self.unaligned_KG2_index = [], []
        for i in self.KG1.desc_index:
            if i not in self.aligned_KG1:
                self.unaligned_KG1_index.append(i)
        self.unaligned_KG1_index = np.array(self.unaligned_KG1_index)
        for i in self.KG2.desc_index:
            if i not in self.aligned_KG2:
                self.unaligned_KG2_index.append(i)
        self.unaligned_KG2_index = np.array(self.unaligned_KG2_index)
        self.KG1.n_ents = len(self.KG1.ents)
        self.KG2.n_ents = len(self.KG2.ents)
        nx.set_node_attributes(self.GoTree, name='connect',values=temp_proDegree)
        self.max_proDegree,self.min_proDegree  = max(temp_proDegree.values()), min(temp_proDegree.values())
        print("Loaded aligned entities from", filename, ". #pairs:", self.n_align)
        print("Protein degree: [max] ", str(self.max_proDegree), " [min] ", str(self.min_proDegree))
        print("Entity and concept enlarged to size: [ entity: " + str(len(self.KG1.ents)) +"] [ concept: "+str(len(self.KG2.ents))+"]")

    def construct_tax_node(self, lowerNode, upperNode):
        newChild = not self.GoTree.has_node(lowerNode)
        newParent = not self.GoTree.has_node(upperNode)
        # add node (without update level)
        if not self.GoTree.has_node(lowerNode):
            self.GoTree.add_node(lowerNode, connect=0, depth=0, height=0)
        if not self.GoTree.has_node(upperNode):
            self.GoTree.add_node(upperNode, connect=0, depth=0, height=0)
        # add edge (ToChild only)
        # GoTree.add_edge(lowerNode, upperNode, relname="ToParent")
        self.GoTree.add_edge(upperNode, lowerNode, relname="ToChild")
        temp_height = nx.get_node_attributes(self.GoTree, name='height')
        temp_depth = nx.get_node_attributes(self.GoTree, name='depth')
        # udpate topLevel and btmLevel
        if newParent and newChild: # case 1
            temp_height[lowerNode], temp_height[upperNode] = 0,1
            temp_depth[lowerNode], temp_depth[upperNode] = 1,0
        elif not newParent and newChild: # case 2: child is new
            temp_height[lowerNode] = 0 
            temp_depth[lowerNode] = temp_depth[upperNode] + 1
            # update height of all parents
            temp_height = updateUp_height(self.GoTree, upperNode, temp_height)
        elif newParent and not newChild: # case 3: parent is new
            temp_depth[upperNode] = 0
            temp_height[upperNode] =  temp_height[lowerNode] + 1
            # update depth of all children
            temp_depth = updateDown_depth(self.GoTree, lowerNode, temp_depth)
        else: # only add edges
            # nothing needs to be changed at current rules
            pass
        # reset height and depth
        nx.set_node_attributes(self.GoTree, name='height',values=temp_height)
        nx.set_node_attributes(self.GoTree, name='depth',values=temp_depth)

    def load_taxonomy(self, filename,splitter = '@@@', line_end = '\n'):
        taxonomy = []
        last_c = -1
        last_r = -1
        self.n_taxonomy = 0
        self.taxonomy = []
        # before load hierarchy
        for i in range(len(self.KG2.ents)): # avoid index error
            self.GoTree.add_node(i, connect=0, depth=0, height=0)
        #for line in open(filename):
        for lineno, line in tqdm(enumerate(open(filename)), desc='load GoTerm Structures', unit=' struct triples'):
            line = line.rstrip(line_end).split(splitter)
            c1 = self.KG2.ent_str2index(line[0])
            c2 = self.KG2.ent_str2index(line[2])
            # ===== Expand Taxonomy =====
            if c1 == None:
                current_entnum = len(self.KG2.ents)
                self.KG2.ents[current_entnum] = line[0]
                self.KG2.index_ents[line[0]] = current_entnum
                self.KG2.ent_tokens[current_entnum] = set(line[0].replace('(','').replace(')','').split(' '))
                c1 = self.KG2.ent_str2index(line[0])
            if c2 == None:
                current_entnum = len(self.KG2.ents)
                self.KG2.ents[current_entnum] = line[2]
                self.KG2.index_ents[line[2]] = current_entnum
                self.KG2.ent_tokens[current_entnum] = set(line[2].replace('(','').replace(')','').split(' '))
                c2 = self.KG2.ent_str2index(line[2])

            # ====== Construct Taxonomy Tree ===
            self.construct_tax_node(c1, c2) # parameter: lowerNode, upperNode
            # ====== End Of Preprocessing ======

            self.taxonomy.append((c1,c2))
            if self.con12.get(c1) == None:
                self.con12[c1] = set([c2])
            else:
                self.con12[c1].add(c2)
            if self.con21.get(c2) == None:
                self.con21[c2] = set([c1])
            else:
                self.con21[c2].add(c1)
            self.n_taxonomy += 1
        self.taxonomy = np.array(self.taxonomy)
        print("Loaded onto taxonomy from", filename, ". #pairs:", self.n_taxonomy)
        print("Entity and concept enlarged to size: [ entity: " + str(len(self.KG1.ents)) +"] [ concept: "+str(len(self.KG2.ents))+"]")
        
    def load_valid(self, filename, size=1024, lan1 = 'en', lan2 = 'fr', splitter = '@@@', line_end = '\n', desc=False):
        '''Load the dataset.'''
        self.align_valid = []
        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            e1 = self.KG1.ent_str2index(line[0])
            e2 = self.KG2.ent_str2index(line[1])
            if e1 == None or e2 == None:
                continue
            if self.ent12.get(e1) == None:
                self.ent12[e1] = set([e2])
            else:
                self.ent12[e1].add(e2)
            if self.ent21.get(e2) == None:
                self.ent21[e2] = set([e1])
            else:
                self.ent21[e2].add(e1)
            if (not self.KG1.get_desc_embed(e1) is None) and (not self.KG2.get_desc_embed(e2) is None):
                self.align_valid.append((e1, e2))
                if len(self.align_valid) >= size:
                    break
        self.align_valid = np.array(self.align_valid)
        print("Loaded validation entities from", filename, ". #pairs:", size)

    def load_more_gt(self, filename):
        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            e1 = self.KG1.ent_str2index(line[0])
            e2 = self.KG2.ent_str2index(line[1])
            if e1 == None or e2 == None:
                continue
            if self.ent12.get(e1) == None:
                self.ent12[e1] = set([e2])
            else:
                self.ent12[e1].add(e2)
            if self.ent21.get(e2) == None:
                self.ent21[e2] = set([e1])
            else:
                self.ent21[e2].add(e1)
            print("Loaded more gt file for negative sampling from", filename)

    def num_align(self):
        return self.n_align

    def num_taxonomy(self):
        return self.n_taxonomy
    
    def num_align_desc(self):
        '''Returns number of entities. 

        This means all entities have index that 0 <= index < num_ents().
        '''
        return self.n_align_desc
 
    def corrupt_desc_pos(self, align, pos, sample_global=True):
        assert (pos in [0, 1])
        hit = True
        res = None
        while hit:
            res = np.copy(align)
            if pos == 0:
                if sample_global:
                    samp = np.random.choice(self.KG1.desc_index)
                else:
                    samp = np.random.choice(self.aligned_KG1_index)
                if samp not in self.ent21[align[1]]:
                    hit = False
                    res = np.array([samp, align[1]])
            else:
                if sample_global:
                    samp = np.random.choice(self.KG2.desc_index)
                else:
                    samp = np.random.choice(self.aligned_KG2_index)
                if samp not in self.ent12[align[0]]:
                    hit = False
                    res = np.array([align[0], samp])
        return res

    def corrupt_desc(self, align, tar=None):
        pos = tar
        if pos == None:
            pos = np.random.randint(2)
        return self.corrupt_desc_pos(align, pos)
    
    def corrupt_align_pos(self, align, pos):
        assert (pos in [0, 1])
        hit = True
        res = None
        while hit:
            res = np.copy(align)
            if pos == 0:
                samp = np.random.randint(self.KG1.num_ents())
                if samp not in self.ent21[align[1]]:
                    hit = False
                    res = np.array([samp, align[1]])
            else:
                samp = np.random.randint(self.KG2.num_ents())
                if samp not in self.ent12[align[0]]:
                    hit = False
                    res = np.array([align[0], samp])
        return res

    def corrupt_tax_pos(self, align, pos):
        assert (pos in [0, 1])
        hit = True
        res = None
        while hit:
            res = np.copy(align)
            if pos == 0:
                samp = np.random.randint(self.KG2.num_ents())
                if samp not in self.con21[align[1]]:
                    hit = False
                    res = np.array([samp, align[1]])
            else:
                samp = np.random.randint(self.KG2.num_ents())
                if samp not in self.con12[align[0]]:
                    hit = False
                    res = np.array([align[0], samp])
        return res

    def corrupt_align(self, align, tar=None):
        pos = tar
        if pos == None:
            pos = np.random.randint(2)
        return self.corrupt_align_pos(align, pos)

    def corrupt_tax(self, align, tar=None):
        pos = tar
        if pos == None:
            pos = np.random.randint(2)
        return self.corrupt_tax_pos(align, pos)
    
    #corrupt 
    def corrupt_desc_batch(self, a_batch, tar = None):
        np.random.seed(int(time.time()))
        return np.array([self.corrupt_desc(a, tar) for a in a_batch])

    def corrupt_align_batch(self, a_batch, tar = None):
        np.random.seed(int(time.time()))
        return np.array([self.corrupt_align(a, tar) for a in a_batch])

    def corrupt_tax_batch(self, a_batch, tar = None):
        np.random.seed(int(time.time()))
        return np.array([self.corrupt_tax(a, tar) for a in a_batch])
    
    def sample_false_pair(self, batch_sizeA):
        a = np.random.choice(self.unaligned_KG1_index, batch_sizeA)
        b = np.random.choice(self.unaligned_KG2_index, batch_sizeA)
        return np.array([(a[i], b[i]) for i in range(batch_sizeA)])
    
    def expand_align(self, list_of_pairs):
        # TODO
        pass
    
    def token_overlap(self, set1, set2):
        min_len = min(len(set1), len(set2))
        hit = 0.
        for tk in set1:
            for tk2 in set2:
                if tk == tk2:
                    hit += 1
        return hit / min_len

    def save(self, filename):
        f = open(filename,'wb')
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print("Save data object as", filename)
    def load(self, filename):
        f = open(filename,'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print("Loaded data object from", filename)