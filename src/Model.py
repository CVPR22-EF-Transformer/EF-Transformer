import torch
import torch.nn as nn
import torch.nn.functional as F

from Layers import *
from Utils import *

class EFTransformer(nn.Module):
    def __init__(self, args):
        super(EFTransformer, self).__init__()
        self.inDim         = args.inChannelsQ
        self.dimPred       = args.dimPred
        self.predFlag      = args.predFlag

        self.FeatureFusion = FeatureFusion(args)
        self.Encoder       = STEncoder(args)
        self.Decoder       = STDecoder(args)
        self.Prediction    = Prediction(args)

        self.seen = 0

    def forward(self, sFeatOb, sFeatTa, gFeatOb, gFeatTa ):
        (nB,nPob,nF,nDS) = sFeatOb.size()
        (nB,nPta,nF,nDG) = gFeatTa.size()
        
        featOb = self.FeatureFusion( sFeatOb, gFeatOb, range(nF) ) 
        featTa = self.FeatureFusion( sFeatTa, gFeatTa, range(nF) ) 
        nD = featOb.data.size(3)

        # ---------- Encoder ---------- #
        attFeatOb = self.Encoder(featOb)
                   
        # attFeatTaList = []
        gPred = torch.tensor([]).cuda()
        sPred = torch.tensor([]).cuda()
        featTaIn = featTa.index_select(2,torch.LongTensor([0]).cuda()).view(nB,nPta,1,nD)

        for fId in range(1,nF):  
            # ---------- Decoder ---------- #
            featObt   = attFeatOb.index_select(2,torch.LongTensor([fId]).cuda()).view(nB,nPob,1,nD)
            featTat   = self.Decoder(featObt,featTaIn)
            # attFeatTaList.append(featTat)

            # --------- Prediction -------- #
            predt    = self.Prediction(featTat)
            gPred    = torch.cat((gPred,predt[0]),2).view(nB,nPta,fId,self.dimPred)  
            gFeatTat = torch.cat((predt[0],torch.tensor([float(fId)/10]).cuda().repeat(nB,nPta,1,1)),3)

            if self.predFlag[1] == 1:
                sPred    = torch.cat((sPred,predt[1]),2).view(nB,nPta,fId,nDS)  
                sFeatTat = predt[1].view(nB,nPta,1,nDS)
            else:
                sFeatTat = sFeatTa.index_select(2,torch.LongTensor([fId]).cuda())

            featTat  = self.FeatureFusion(sFeatTat,gFeatTat,[fId])
            featTaIn = torch.cat((featTaIn,featTat),2).view(nB,nPta,fId+1,nD)  

        # -------- Loss Function ------ #
        gOutput, gGt = self.Prediction.PrepareG(gPred,gFeatTa.detach())        
        gLoss, gStat = self.LossG( gOutput, gGt )

        if self.predFlag[1] == 1:
            sGt = sFeatTa.index_select(2,torch.LongTensor(range(1,nF)).cuda())
            sLoss, sStat = self.LossS( sPred, sGt )
        else:
            sLoss = torch.tensor([0.]).cuda()
            sStat = [0.,0.]

        return gPred, sPred, gLoss, sLoss, gStat, sStat



    def LossG(self, pred, gt ):
        (nB,nP,nF,nD) = gt.data.size() 

        gt = gt.detach()
        loss = nn.MSELoss(size_average=False)(pred, gt)/2.0

        # statistic
        dist  = EuclideanDistance(pred,gt,3,2,3,1.).view(nB*nP,nF) # Use Type 2 to compute stats
        nMiss = [ nB*nP*nF-torch.sum(dist<=0.01).tolist(), nB*nP*nF-torch.sum(dist<=0.03).tolist(), nB*nP*nF-torch.sum(dist<=0.05).tolist() ]
        maxDist, idx = torch.max(dist, dim=0 )
        stat  = [nMiss, maxDist.view(nF).data]
        
        return loss, stat


    def LossS(self, pred, gt ):
        (nB,nP,nF,nD) = gt.data.size() 
        mask = torch.ones(nB,nP,nF,nD).cuda() * 1.0
        mask[gt>0] = 2.0

        gt = gt.detach()
        mask = mask.detach()
        loss = nn.MSELoss(size_average=False)(pred*mask, gt*mask)/2.0

        # statistic
        nPos = torch.sum( gt >  0 )
        nNeg = torch.sum( gt == 0 )
        avgPos = torch.sum( pred[gt> 0]) / nPos.float()
        avgNeg = torch.sum( pred[gt==0]) / nNeg.float()
        stat = [ avgPos, avgNeg ]

        return loss, stat













































