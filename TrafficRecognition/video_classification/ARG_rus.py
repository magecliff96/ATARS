import torch 
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np
import sys
sys.path.append('/media/hcis-s19/DATA/Action-Slot/scripts')
from utils import *
from base_model import Object_based
from classifier import Head

def calc_pairwise_distance_3d(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [B,N,D]
        Y: [B,M,D]
    Returns:
        dist: [B,N,M] matrix of euclidean distances
    """
    B=X.shape[0]
    
    rx=X.pow(2).sum(dim=2).reshape((B,-1,1))
    ry=Y.pow(2).sum(dim=2).reshape((B,-1,1))
    
    dist=rx-2.0*X.matmul(Y.transpose(1,2))+ry.transpose(1,2)
    
    return torch.sqrt(dist)

class GCN_Module(nn.Module):
    def __init__(self, NFR, NG, NFG):
        super(GCN_Module, self).__init__()
        
        self.pos_threshold = 0.2
        self.NFR = NFR
        self.NG = NG
        self.NFG = NFG
        NFG_ONE=NFG
        
        
        self.fc_rn_theta_list=torch.nn.ModuleList([ nn.Linear(NFG,NFR) for i in range(NG) ])
        self.fc_rn_phi_list=torch.nn.ModuleList([ nn.Linear(NFG,NFR) for i in range(NG) ])
        
        
        self.fc_gcn_list=torch.nn.ModuleList([ nn.Linear(NFG,NFG_ONE,bias=False) for i in range(NG) ])
        
        self.nl_gcn_list=torch.nn.ModuleList([ nn.LayerNorm([NFG_ONE]) for i in range(NG) ])
        
            

        
    def forward(self,graph_boxes_features,boxes_in_flat=None,OW=None):
        B,N = graph_boxes_features.shape[:2]
        
        pos_threshold=self.pos_threshold
        
        
        
        relation_graph=None
        graph_boxes_features_list=[]
        for i in range(self.NG):
            graph_boxes_features_theta=self.fc_rn_theta_list[i](graph_boxes_features)  #B,N,NFR
            graph_boxes_features_phi=self.fc_rn_phi_list[i](graph_boxes_features)  #B,N,NFR

            similarity_relation_graph=torch.matmul(graph_boxes_features_theta,graph_boxes_features_phi.transpose(1,2))  #B,N,N

            similarity_relation_graph=similarity_relation_graph/np.sqrt(self.NFR)

            similarity_relation_graph=similarity_relation_graph.reshape(-1,1)  #B*N*N, 1

            relation_graph=similarity_relation_graph

            relation_graph = relation_graph.reshape(B,N,N)

            relation_graph = torch.softmax(relation_graph,dim=2)       

            one_graph_boxes_features=self.fc_gcn_list[i]( torch.matmul(relation_graph,graph_boxes_features) )  #B, N, NFG_ONE
            one_graph_boxes_features=self.nl_gcn_list[i](one_graph_boxes_features)
            one_graph_boxes_features=F.relu(one_graph_boxes_features)
            
            graph_boxes_features_list.append(one_graph_boxes_features)
        
        graph_boxes_features=torch.sum(torch.stack(graph_boxes_features_list),dim=0) #B, N, NFG
        
        return graph_boxes_features,relation_graph

class ARG(Object_based):
    """
    main module of base model for the volleyball
    """
    def __init__(
        self,
        args,
        max_N=0,
        NFB=512,
        K=3,
        num_actor_class=None,
        gcn_layers=1,
        NFR=256,
        ):
        super().__init__(args,K,NFB,max_N)
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        self.NFR = NFR
        
        self.head = Head(NFB, num_actor_class)
        
        self.gcn_list = torch.nn.ModuleList([GCN_Module(NFR, 16, NFB)  for _ in range(gcn_layers) ])    
        
        self.dropout_global=nn.Dropout(p=0.3)
        
    def forward(self, x, box=False):
        """
        Args:
            box : b,t,N,4 ; dim: b,c,t,h,w
        """
        T = x[0].shape[1]
        B = len(x)
        if not isinstance(box,list):
            box = box.reshape(-1,self.max_N,4)
            box = list(box)   
        assert len(box) == B*T
        
        features = self.extract_features(x) # b,d,t,H,W

        obj_features = self.get_object_features(features,box)
        
        graph_boxes_features = obj_features.reshape(B,T*(self.max_N),self.NFB)
        
        for i in range(len(self.gcn_list)):
            graph_boxes_features, relation_graph = self.gcn_list[i](graph_boxes_features)
        
        graph_boxes_features = graph_boxes_features.reshape(B,T,self.max_N,self.NFB)  
        obj_features = obj_features.reshape(B,T,self.max_N,self.NFB)
        
        boxes_states = graph_boxes_features+obj_features
    
        boxes_states = self.dropout_global(boxes_states) # b,t,N,NFB
        

        y_actor = self.head(boxes_states[:,:,1:])
        y_actor = y_actor.mean(dim=1) # b,N,class
        y_actor, _ = y_actor.max(dim=1) #b, class
        return y_actor
        
