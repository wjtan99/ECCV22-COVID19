"""
Created on Sun Apr 19 23:11:35 2020

@author: esat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from .BERT.bert import BERT, BERT2, BERT3, BERT4, BERT5, BERT6

from .r2plus1d import r2plus1d_34_32_ig65m, r2plus1d_34_32_kinetics, flow_r2plus1d_34_32_ig65m

from .representation_flow import resnet_50_rep_flow


__all__ = ['rgb_r2plus1d_32f_34', 'rgb_r2plus1d_32f_34_bert10', 'rgb_r2plus1d_64f_34_bert10', 'feature_bert10']


class feature_bert10(nn.Module):
    def __init__(self, num_classes, length, modelPath=''):
        super(feature_bert10, self).__init__()

        self.hidden_size=512
        self.n_layers=2
        self.attn_heads=4 #was 2, 4 
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.7)       
        self.act = nn.ReLU()
 
        self.bert = BERT5(self.hidden_size, self.length , hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))

        self.fc_action = nn.Linear(self.hidden_size, 128)
        self.fc_action1 = nn.Linear(128, 32)
        self.fc_action2 = nn.Linear(32, num_classes)

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc_action1.weight)
        self.fc_action1.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc_action2.weight)
        self.fc_action2.bias.data.zero_()
        
    def forward(self, x):

        #print("x0.shape = ", x.shape) 
        x = x.view(x.size(0), self.hidden_size, self.length)
        #print("x1.shape = {}".format(x.shape))      
        x = x.transpose(1,2)

        input_vectors=x
        norm = input_vectors.norm(p=2, dim = -1, keepdim=True)
        input_vectors = input_vectors.div(norm)

        output, maskSample = self.bert(x) #x 

        classificationOut = output[:,0,:]
        #norm = classificationOut.norm(p=2, dim = -1, keepdim=True)
        #classificationOut = classificationOut.div(norm)

        #print("x4.shape = {}".format(classificationOut.shape))      
        sequenceOut=output[:,1:,:]
        norm = sequenceOut.norm(p=2, dim = -1, keepdim=True)
        sequenceOut = sequenceOut.div(norm)

        output=self.dp(classificationOut)
        x = self.fc_action(output)  
        
        x = self.dp(x) 
        x = self.fc_action1(x)
        x = self.dp(x) 
        x = self.fc_action2(x)
         
        return x, input_vectors, sequenceOut, maskSample, classificationOut 
  



class rgb_r2plus1d_32f_34(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_r2plus1d_32f_34, self).__init__()
        self.num_classes=num_classes
        self.dp = nn.Dropout(p=0.8)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.features=nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])
        
        #self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.fc_action = nn.Linear(512, num_classes)
        for param in self.features.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = self.fc_action(x)
        return x,x,x,x,x
    
    def mars_forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class rgb_r2plus1d_64f_34(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_r2plus1d_64f_34, self).__init__()
        self.num_classes=num_classes
        self.dp = nn.Dropout(p=0.8)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.features=nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=False, progress=True).children())[:-2])
        
        #self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.fc_action = nn.Linear(512, num_classes)
        for param in self.features.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = self.fc_action(x)
        return x,x,x,x,x
    
    def mars_forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    
class rgb_r2plus1d_32f_34_bert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_r2plus1d_32f_34_bert10, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)        

        self.avgpool = nn.AvgPool3d((1, 14, 14), stride=1) #7,7 for 112x112 

        self.features=nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])        
        self.bert = BERT5(self.hidden_size, 4 , hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
        for param in self.features.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):

        #print("input.shape = ", x.shape) 

        x = self.features(x)

        #print("x1.shape = {}".format(x.shape)) 

        x = self.avgpool(x)
        #print("x2.shape = {}".format(x.shape))        

        x = x.view(x.size(0), self.hidden_size, 4)
        #print("x3.shape = {}".format(x.shape))      

        x = x.transpose(1,2)
        input_vectors=x
        norm = input_vectors.norm(p=2, dim = -1, keepdim=True)
        input_vectors = input_vectors.div(norm)

        #print("bert input.shape =", x.shape)  
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]

        #print("x4.shape = {}".format(classificationOut.shape))      

        sequenceOut=output[:,1:,:]
        norm = sequenceOut.norm(p=2, dim = -1, keepdim=True)
        sequenceOut = sequenceOut.div(norm)
        #print("x5.shape = {}".format(sequenceOut.shape))    

        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample, classificationOut 
    
    
class rgb_r2plus1d_64f_34_bert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_r2plus1d_64f_34_bert10, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)

        self.features=nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])        
        self.bert = BERT5(self.hidden_size, 8 , hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
        for param in self.features.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), self.hidden_size, 8)
        x = x.transpose(1,2)
        input_vectors=x
        norm = input_vectors.norm(p=2, dim = -1, keepdim=True)
        input_vectors = input_vectors.div(norm)
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        norm = sequenceOut.norm(p=2, dim = -1, keepdim=True)
        sequenceOut = sequenceOut.div(norm)
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample, classificationOut
    
    
    
    
