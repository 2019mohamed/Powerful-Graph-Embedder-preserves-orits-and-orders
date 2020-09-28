
import os.path as osp
import torch

import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, DeepGraphInfomax, ChebConv

from sklearn.metrics.cluster import (v_measure_score, homogeneity_score,
                                     completeness_score)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
from networkx.readwrite.edgelist import read_edgelist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans , SpectralClustering , AgglomerativeClustering
from sklearn.metrics import accuracy_score 
import numpy as np
from munkres import Munkres, print_matrix
from torch_geometric.data import Data , DataLoader


def readgraph(dataset = ''):
    G = read_edgelist('C:\\Users\\M\\'+str(dataset)+'.txt') 
    #G = nx.generators.random_graphs.erdos_renyi_graph(179,0.5)
    edges1 = []
    edges2 = []
    
    edges = list(G.edges())
    for pair in edges:
        u , v = int(pair[0]) , int(pair[1])
        edges1.append(u)
        edges1.append(v)
        
        edges2.append(v)
        edges2.append(u)
        
    
    lines = []
    os.chdir('C:\\Users\\M\\Desktop\\orbit-counting\\wrappers')
    os.system('python orbit_counts.py C:\\Users\\M\\'+str(dataset)+'.edges 5 -i -c' )
    with open('C:\\Users\\M\\Desktop\\orbit-counting\\wrappers\\induced_counts_out.txt','r') as reader:
        lines = reader.readlines()
    
    features = []
    
    node_feature = []
    i = 0
    
    #if dataset == 'testgraph':
    def floatoint (n):
        return int(str(n).replace('.000000',''))
    
    #dis = set()
    listoFtruelable = []
    for line in lines:
        node_feature = line.split(sep= ' ')
        node_feature = node_feature[0:3]
        features.append([])
        for num in node_feature:
            if num == '\n' or num == '\t\n'or num == '.':
                continue
            features[i].append(float(num))
            #dis.add(float(num))
        i = i+1
        
    lines = []
    #C:\Users\M\8eularorbits
    with open('C:\\Users\\M\\'+str(dataset)+'orbit.txt','r')as reader:
        lines = reader.readlines()
    intlines = []
    i = 0
    for line in lines:
        node_feature = line.split(sep = ' ')
        for n in node_feature:
            if n == '' :
                continue
            intlines.append([int(n)])
            listoFtruelable.append(int(n))
        i = i+1
    true_label = intlines
    
    #sim = nx.simrank_similarity(G)
    #target = [[sim[u][v] for v in sim[u]] for u in sim]
    return features , edges1 , edges2 , true_label , listoFtruelable 


print('Here We Gooo...')

f , e1,e2 , true_label , label = readgraph(dataset = 'cora')
#print(f)
#print(true_label)
edge_index = torch.tensor([e1,e2] , dtype=torch.long)
x = torch.tensor(f, dtype=torch.float)
y = torch.tensor(true_label , dtype=torch.long)

data = Data(x=x, edge_index=edge_index , y =y)

print('Data is now prepared !!')

class GCNLayer(nn.Module):
    def __init__(self , in_channels , out_channels ):
        super(GCNLayer, self).__init__()
        self.gcn0 = GCNConv(in_channels , out_channels )
        
        self.relu0 = nn.ReLU(out_channels)
        '''
        self.gcn1 = GCNConv(64 , 32)
        self.relu1 = nn.ReLU(32)
        self.gcn2 = GCNConv(32 , out_channels )
        self.relu2 = nn.ReLU(out_channels)
        '''
    def forward(self,x ,edge_index):
        x = self.relu0(self.gcn0(x,edge_index))
        #x = self.relu1(self.gcn1(x,edge_index))
        #x = self.relu2(self.gcn2(x,edge_index))

        return x
    
    def loss (self, embeddings, target):
        predict = torch.mm(embeddings, embeddings.t())
        target = torch.mm(target, target.t())
        reconstruction_loss = torch.nn.MSELoss()(predict, target)
        return reconstruction_loss



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNLayer(data.num_features , 64).to(device)

data = data.to(device)
X = data.x
edges = data.edge_index

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train ():
    model.train()
    emb = model(X,edges)
    loss = model.loss(emb,x)
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    z= model(X, edges)
    out0 = PCA(n_components=2).fit_transform(z.detach().numpy())
    plt.scatter(out0[:,0],out0[:,1])
    plt.show()
    clustering_pred = AgglomerativeClustering(n_clusters = len(set(label))).fit(out0).labels_
    # Compute metrics
    
    
    pred_label = clustering_pred
    true_label = label
    l1 = list(set(true_label))
    numclass1 = len(l1)
    
    l2 = list(set(pred_label))
    numclass2 = len(l2)
    if numclass1 != numclass2:
        print('Class Not equal, Error!!!!')
        
    
    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(true_label) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if pred_label[i1] == c2]
            cost[i][j] = len(mps_d)
    
            # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    
    indexes = m.compute(cost)
    #idx = indexes[2][1]
            # get the match results
    new_predict = np.zeros(len(pred_label))
    for i, c in enumerate(l1):
                # correponding label in l2:
        c2 = l2[indexes[i][1]]
                # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(pred_label) if elm == c2]
        new_predict[ai] = c
    print('ACC ', accuracy_score(true_label, new_predict))
    
    
for epoch in range(1, 300):
    loss = train()
    print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))

acc = test()


