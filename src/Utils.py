import torch
import math
import time
import random
import numpy as np


def Convert2CPU(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def EuclideanDistance(pointList0, pointList1, dimIdx, dimX, dimY, r):
    x0 = pointList0.index_select(dimIdx,torch.LongTensor([dimX]).cuda())
    x1 = pointList1.index_select(dimIdx,torch.LongTensor([dimX]).cuda())
    y0 = pointList0.index_select(dimIdx,torch.LongTensor([dimY]).cuda())
    y1 = pointList1.index_select(dimIdx,torch.LongTensor([dimY]).cuda())
    dist = torch.sqrt( (x0-x1) * (x0-x1) + (y0-y1) * (y0-y1) * r * r )
    return dist 

def ObvTarSplit(args, sFeat, gFeat, phase):   
    (nB,nP,nF,nDS) = sFeat.data.size()
    (nB,nP,nF,nDG) = gFeat.data.size()

    if args.dataset == 'Tennis' and phase == 'train':
        tarPos = [random.randint(0,1)]
    else:
        tarPos = args.tarPos

    obvPos = [ i for i in range(nP) if i not in tarPos ]
    nPob = len(obvPos)
    nPta = len(tarPos)

    sFeatOb = sFeat[:,obvPos,:,:].view(nB,nPob,nF,nDS)
    sFeatTa = sFeat[:,tarPos,:,:].view(nB,nPta,nF,nDS)
    gFeatOb = gFeat[:,obvPos,:,:].view(nB,nPob,nF,nDG)
    gFeatTa = gFeat[:,tarPos,:,:].view(nB,nPta,nF,nDG)

    return sFeatOb, sFeatTa, gFeatOb, gFeatTa

def BuildMask(l,k):
    mask1 = torch.tensor(range(l)).view(1,l).repeat(l,1)
    mask2 = torch.tensor(range(l)).view(l,1).repeat(1,l)
    mask  = ( ( mask2 - mask1 + k ) > 0 )
    return mask

def LengthClassify(traj, dataset):
    if dataset == 'Dance':
        thres = [0,0.1,0.2,10000.]
    else:
        thres = [0,0.09,0.18,10000.]

    (nC,nP,nF,nD) = traj.size()

    traj = traj - torch.cat((torch.zeros(nC,nP,1,nD),traj[:,:,:nF-1,:]),2)
    ddxy = torch.sqrt( traj[:,:,:,0] * traj[:,:,:,0] + traj[:,:,:,1] * traj[:,:,:,1] * 9.0/16.0 * 9.0/16.0 ) 
    gtLength = ddxy.sum(2) # nClip x nP

    gtLengthIdx = torch.zeros(nC,nP) - 1
    gtLengthCount = torch.zeros(3)
    for cId in range(nC):
        for pId in range(nP):
            for l in range(3):
                if gtLength[cId,pId] > thres[l] and gtLength[cId,pId] <= thres[l+1]:
                    gtLengthIdx[cId,pId] = l
                    gtLengthCount[l] += 1

    return gtLengthIdx, gtLengthCount

def macroF1(cm):
    nC = cm.shape[0]
    f1 = 0.
    for cId in range(nC):
        rec = cm[cId,cId] / float(cm[cId,:].sum())
        pre = cm[cId,cId] / float(cm[:,cId].sum())
        f1  = f1 + 2 * rec * pre / ( rec + pre )
    return f1 / nC 

