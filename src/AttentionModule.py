import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import init

class Attention(nn.Module):
    def __init__(self, dropout):
        super(Attention, self).__init__()    
        self.dropout = dropout
        self.DropoutLayer = nn.Dropout(dropout)
        
    def forward(self,qX,kX,vX,mask=None):
        (nB,nQ,nDq) = qX.data.size()
        (nB,nDk,nK) = kX.data.size()
        (nB,nV,nDv) = vX.data.size()

        attMat = torch.matmul(qX,kX)
        if mask is not None:
            attMat = attMat.masked_fill(mask == 0, -1e9)
        attMat = F.softmax( attMat, dim=-1 ) / (nDk**0.5)             # (nB,nQ,nK) 
        if self.dropout > 0:
            attMat = self.DropoutLayer(attMat)
        attX = torch.matmul(attMat,vX).permute(0,2,1).contiguous().view(nB,nDv,nQ) # (nB,nDv,nQ)        

        return attX

class MulitiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MulitiHeadAttention, self).__init__()
        self.inDimQ     = args.inChannelsQ
        self.inDimK     = args.inChannelsK
        self.inDimV     = args.inChannelsV
        self.interDimQK = args.interChannelsQK
        self.interDimV  = args.interChannelsV
        self.nHead      = args.nHead
        self.dropout    = args.dropout
        self.oneDimQK   = int(self.interDimQK/self.nHead)
        self.oneDimV    = int(self.interDimV/self.nHead)

        if self.interDimQK % self.nHead != 0 or self.interDimV % self.nHead != 0:
            print "interDimQK(%d) / nHead(%d) should be an integer."%(self.interDimQK,self.nHead)
            print "interDimV(%d)  / nHead(%d) should be an integer."%(self.interDimV,self.nHead)
            tmp = raw_input()

        self.create_network()

    def create_network(self):
        self.WQ = nn.Conv2d(self.inDimQ, self.interDimQK, 1, 1, 0)
        self.WK = nn.Conv2d(self.inDimK, self.interDimQK, 1, 1, 0)
        self.WV = nn.Conv2d(self.inDimV, self.interDimV,  1, 1, 0)

        self.WOUT = nn.Conv2d(self.interDimV, self.inDimQ, 1, 1, 0)
        self.Attention = Attention(self.dropout)
        self.DropoutLayer = nn.Dropout(self.dropout)

        torch.nn.init.xavier_uniform_(self.WQ.weight)
        torch.nn.init.xavier_uniform_(self.WK.weight)
        torch.nn.init.xavier_uniform_(self.WV.weight)
        torch.nn.init.xavier_uniform_(self.WOUT.weight)

    def forward(self,qXin,kXin,vXin,mask=None):
        (nB,nDq,nHq,nWq) = qXin.data.size()
        (nB,nDk,nHk,nWk) = kXin.data.size()
        (nB,nDk,nHv,nWv) = vXin.data.size()
        nH    = self.nHead
        nDIqk = self.oneDimQK
        nDIv  = self.oneDimV
        if nHk*nWk % nHv*nWv != 0:
            print "Samples in Key (%d) and Value(%d) should be Equal."%(nHk*nWk,nHv*nWv)
            tmp = raw_input()

        qX = self.WQ(qXin).view(nB*nH,nDIqk,nHq*nWq).permute(0,2,1)                                 # (nB*nH,nHq*nWq,nDI)
        kX = self.WK(kXin).view(nB*nH,nDIqk,nHk*nWk)                                                # (nB*nH,nDI,nHk*nWk)
        vX = self.WV(vXin).view(nB*nH,nDIv ,nHv*nWv).permute(0,2,1)                                 # (nB*nH,nHv*nWv,nDI)

        if mask is not None:
            nN = mask.size(0)
            mask  = mask.view(1,nN,nN).repeat(nB*nH,1,1).cuda()
        attX = self.Attention(qX,kX,vX,mask).view(nB,nH,nDIv,nHq*nWq).view(nB,nH*nDIv,nHq,nWq)
        attX = self.WOUT(attX).view(nB,nDq,nHq,nWq)
        if self.dropout > 0:
            attX = self.DropoutLayer(attX)
        
        return attX