import torch
import torch.nn as nn
import numpy as np

from Utils import *
from AttentionModule import MulitiHeadAttention

class FeatureFusion(nn.Module):
    def __init__(self, args):
        super(FeatureFusion, self).__init__()
        self.inDim       = args.inChannelsQ
        self.clipLength  = args.clipLength
        self.oriDimS     = args.oriDimS
        self.oriDimG     = args.oriDimG
        self.dimS        = args.dimS
        self.dimG        = args.dimG
        self.peMat       = self.PositionalEncoding(self.inDim,self.clipLength).cuda()
        self.MappingS    = nn.Conv2d(self.oriDimS, self.dimS, 1, 1, 0)
        self.MappingG    = nn.Conv2d(self.oriDimG, self.dimG, 1, 1, 0)

    def PositionalEncoding(self, D, T, nSplit=1 ):
        D = int(D/nSplit)
        pe = torch.zeros(T,D).cuda()
        for t in range(T):
            ft = torch.tensor(t).float()
            for d in range(D):
                fd = torch.tensor(d).float()
                if d % 2 == 0:
                    pe[t,d] = torch.sin( t / torch.pow( 10000, fd / D ) )
                else:
                    pe[t,d] = torch.cos( t / torch.pow( 10000, fd / D ) )
        return pe

    def forward(self, sFeat, gFeat, fIndex):
        (nB,nP,nF,nDS) = sFeat.data.size()
        nDG = gFeat.data.size(3)
        gmFeat = self.MappingG(gFeat.contiguous().view(nB*nP*nF,nDG,1,1)).view(nB,nP,nF,self.dimG)
        smFeat = self.MappingS(sFeat.contiguous().view(nB*nP*nF,nDS,1,1)).view(nB,nP,nF,self.dimS)
        feat = torch.cat((gmFeat,smFeat),3).view(nB,nP,nF,self.dimG+self.dimS)
        peFeat = self.peMat.cuda().view(1,1,self.clipLength,self.inDim).repeat(nB,nP,1,1).view(nB,nP,self.clipLength,self.inDim)
        feat = feat + peFeat.index_select(2,torch.LongTensor(fIndex).cuda()).view(nB,nP,nF,self.inDim)
        # feat = self.MappingNorm(feat)
        return feat

class FFN(nn.Module):
    def __init__(self, args, inDim, outDim):
        super(FFN, self).__init__()
        self.inDim   = inDim
        self.outDim  = outDim
        self.ffnDim  = args.ffnChannels
        self.dropout = args.dropout

        self.FFN          = nn.Sequential(nn.Conv2d(self.inDim,self.ffnDim,1,1,0),
                                          nn.LeakyReLU(0.1, inplace=True),
                                          nn.Conv2d(self.ffnDim,self.inDim,1,1,0))
        self.DropoutLayer = nn.Dropout(self.dropout)
        self.FfnNorm      = nn.LayerNorm(self.inDim)

        nn.init.normal_(self.FfnNorm.weight)
        nn.init.constant_(self.FfnNorm.bias, 0)

    def forward(self,x):
        (nB,nP,nF,nD) = x.data.size()
        ffnX = self.FFN(x.permute(0,3,1,2)).permute(0,2,3,1).contiguous().view(nB,nP,nF,nD)
        if self.dropout > 0:
            ffnX = self.DropoutLayer(ffnX)
        
        ffnX = ffnX + x
        ffnX = self.FfnNorm(ffnX)       
        return ffnX

class EncoderSpatialSelfAttnLayer(nn.Module):
    def __init__(self, args):
        super(EncoderSpatialSelfAttnLayer, self).__init__()
        self.inDim  = args.inChannelsQ
        self.isFFN  = 0

        self.SSelfAttention = MulitiHeadAttention(args)
        self.TSelfAttention = MulitiHeadAttention(args)
        self.FFN            = FFN(args, self.inDim, self.inDim)
        self.AttNorm        = nn.LayerNorm(self.inDim)

        nn.init.normal_(self.AttNorm.weight)
        nn.init.constant_(self.AttNorm.bias, 0)

    def forward(self,x):
        (nB,nP,nF,nD) = x.data.size()

        sX    = x.permute(0,2,3,1).contiguous().view(nB*nF,nD,nP,1)     
        sAttX = self.SSelfAttention(sX,sX,sX).view(nB,nF,nD,nP).permute(0,3,1,2)    

        attX = sAttX + x 
        attX = self.AttNorm(attX)

        if self.isFFN == 1:
            attX = self.FFN(attX)
            
        return attX

class EncoderTemporalSelfAttnLayer(nn.Module):
    def __init__(self, args):
        super(EncoderTemporalSelfAttnLayer, self).__init__()
        self.inDim  = args.inChannelsQ
        self.isFFN  = 1

        self.SSelfAttention = MulitiHeadAttention(args)
        self.TSelfAttention = MulitiHeadAttention(args)
        self.FFN            = FFN(args, self.inDim, self.inDim)
        self.AttNorm        = nn.LayerNorm(self.inDim)
         
        nn.init.normal_(self.AttNorm.weight)
        nn.init.constant_(self.AttNorm.bias, 0)

    def forward(self,x,mask=None):
        (nB,nP,nF,nD) = x.data.size()

        tX    = x.permute(0,1,3,2).contiguous().view(nB*nP,nD,nF,1)     
        tAttX = self.TSelfAttention(tX,tX,tX,mask).view(nB,nP,nD,nF).permute(0,1,3,2) 

        attX = tAttX + x 
        attX = self.AttNorm(attX)

        if self.isFFN == 1:
            attX = self.FFN(attX)
            
        return attX

class DecoderSelfAttnLayer(nn.Module):
    def __init__(self, args):
        super(DecoderSelfAttnLayer, self).__init__()
        self.inDim  = args.inChannelsQ

        self.SelfAttention  = MulitiHeadAttention(args)
        self.SelfAttNorm    = nn.LayerNorm(self.inDim)

        nn.init.normal_(self.SelfAttNorm.weight)
        nn.init.constant_(self.SelfAttNorm.bias, 0)

    def forward(self,x):
        (nB,nP,nF,nD) = x.data.size()
        
        # self-attention in target player
        x    = x.permute(0,3,1,2).contiguous().view(nB,nD,nP,nF)    
        attX = self.SelfAttention(x,x,x).view(nB,nD,nP,nF)  

        attX = attX + x
        attX = self.SelfAttNorm(attX.permute(0,2,3,1)).view(nB,nP,nF,nD)

        return attX       

class DecoderCrossAttnLayer(nn.Module):
    def __init__(self, args):
        super(DecoderCrossAttnLayer, self).__init__()
        self.inDim  = args.inChannelsQ
        self.isFFN  = 1

        self.CrossAttention = MulitiHeadAttention(args)
        self.FFN            = FFN(args, self.inDim, self.inDim)
        self.CrossAttNorm   = nn.LayerNorm(self.inDim)
        self.FfnNorm        = nn.LayerNorm(self.inDim)

        nn.init.normal_(self.CrossAttNorm.weight)
        nn.init.constant_(self.CrossAttNorm.bias, 0)
        nn.init.normal_(self.FfnNorm.weight)
        nn.init.constant_(self.FfnNorm.bias, 0)

    def forward(self,xOb,featTa):
        (nB,nPta,nf,nD) = featTa.data.size() # key and value of cross-attention
        (nB,nPob,nX,nD) = xOb.data.size()    # query of cross-attention
           
        # cross-attention: xob is query, featTa is key and value  
        featXOb = xOb.permute(0,3,1,2).contiguous().view(nB,nD,nPob,nX)                 
        featTa  = featTa.permute(0,3,1,2).contiguous().view(nB,nD,nPta,nf)              
        attXOb  = self.CrossAttention(featXOb,featTa,featTa).view(nB,nD,nPob,nX)        

        attXOb = attXOb + featXOb
        attXOb = self.CrossAttNorm(attXOb.permute(0,2,3,1)).contiguous().view(nB,nPob,nX,nD)

        if self.isFFN == 1:
            attXOb = self.FFN(attXOb)

        return attXOb       


class STEncoder(nn.Module):
    def __init__(self, args):
        super(STEncoder, self).__init__()
        self.nLayer      = args.nLayer
        self.clipLength  = args.clipLength
        self.future      = args.future
        self.mask        = BuildMask(self.clipLength,self.future).cuda()

        self.SpatialEncoderList  = nn.ModuleList()
        self.TemporalEncoderList = nn.ModuleList()
        
        for n in range(self.nLayer):
            self.SpatialEncoderList.append( EncoderSpatialSelfAttnLayer(args) )
            self.TemporalEncoderList.append( EncoderTemporalSelfAttnLayer(args) )

    def forward(self,x):
        (nB,nP,nF,nD) = x.data.size()
        for n in range(self.nLayer):
            x = self.SpatialEncoderList[n](x)    
            x = self.TemporalEncoderList[n](x,mask=self.mask)    

        return x

class STDecoder(nn.Module):
    def __init__(self, args):
        super(STDecoder, self).__init__()
        self.nLayer      = args.nLayer

        self.DecoderSelfAttnList  = nn.ModuleList()
        self.DecoderCrossAttnList = nn.ModuleList()

        for n in range(self.nLayer):
            self.DecoderSelfAttnList.append( DecoderSelfAttnLayer(args) )
            self.DecoderCrossAttnList.append( DecoderCrossAttnLayer(args) )

    def forward(self,featOb,featTa):
        (nB,nPta,nf,nD) = featTa.data.size()    # key and value of cross-attention
        (nB,nPob,nX,nD) = featOb.data.size()    # query of cross-attention

        for n in range(self.nLayer):
            # self-attention in target player
            featTa = self.DecoderSelfAttnList[n](featTa)   
            # cross-attention: xob is query, attFeatTa is key and value  
            featOb = self.DecoderCrossAttnList[n](featOb,featTa)  

        return featOb

class Prediction(nn.Module):
    def __init__(self, args ):
        super(Prediction, self).__init__()
        self.inDim     = args.inChannelsQ
        self.dimPred   = args.dimPred
        self.oriDimS   = args.oriDimS
        self.gsFlag    = args.predFlag
        self.nAll      = 5 if args.dataset == 'Tennis' else 10
        self.nTar      = len(args.tarPos)
        self.nObv      = self.nAll - self.nTar
        
        self.Merge     = nn.Conv2d(self.inDim*self.nObv, self.inDim*self.nTar, 1, 1, 0) 
        self.MergeNorm = nn.LayerNorm(self.inDim*self.nTar)

        if self.gsFlag[0] == 1:
            self.PredG = nn.Sequential(nn.Conv2d(self.inDim, self.inDim/2, 1, 1, 0),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(self.inDim/2, self.inDim/2, 1, 1, 0),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(self.inDim/2, self.dimPred, 1, 1, 0))

        if self.gsFlag[1] == 1:
            self.PredS = nn.Sequential(nn.Conv2d(self.inDim, self.inDim/2, 1, 1, 0),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(self.inDim/2, self.inDim/2, 1, 1, 0),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       nn.Conv2d(self.inDim/2, self.oriDimS, 1, 1, 0))

    def SpatialMerge(self, x ):
        (nB,nP,nF,nDin) = x.data.size()
        x = self.Merge(x.permute(0,3,1,2).contiguous().view(nB,nDin*nP,1,nF)).permute(0,2,3,1)
        x = self.MergeNorm(x)
        x = x.view(nB,nF,self.nTar,nDin).permute(0,2,1,3).view(nB,self.nTar,nF,nDin)
        return x

    def forward(self,x):
        x = self.SpatialMerge(x)
        (nB,nP,nF,nD) = x.data.size() 
        predList = []

        if self.gsFlag[0] == 1:
            pred   = self.PredG(x.permute(0,3,1,2).view(nB,nD,nP,nF)).permute(0,2,1,3).view(nB,nP,nF,self.dimPred)            
            predList.append(pred)
        else:
            predList.append(torch.tensor([]))

        if self.gsFlag[1] == 1:
            pred = self.PredS(x.permute(0,3,1,2).view(nB,nD,nP,nF)).permute(0,2,1,3).view(nB,nP,nF,self.oriDimS)
            pred = torch.softmax(pred,3)
            predList.append(pred)
        else:
            predList.append(torch.tensor([]))

        return predList

    def PrepareG(self,pred,gGt):
        (nB,nP,nF,nD) = pred.data.size() # (nB,nPta,nF-1,6)
                                         # gGt (nB,nPta,nF,7)
        predT1 = pred.index_select(3,torch.LongTensor([0,1]).cuda()) - gGt[:,:,0,0:2].contiguous().view(nB,nP,1,2).repeat(1,1,nF,1)
        predT2 = pred.index_select(3,torch.LongTensor([2,3]).cuda())
        predT3 = pred.index_select(3,torch.LongTensor([4,5]).cuda()) + torch.cat((torch.zeros(nB,nP,1,2).cuda(),predT2.index_select(2,torch.LongTensor(range(nF-1)).cuda())),2)
        pred   = torch.cat((predT1,predT2,predT3),3).view(nB,nP,nF,self.dimPred)

        gGt    = gGt[:,:,1:,2:4].view(nB,nP,nF,1,2).repeat(1,1,1,3,1).view(nB,nP,nF,nD)

        return pred, gGt
