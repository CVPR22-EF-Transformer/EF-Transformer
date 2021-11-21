import sys
import os
import time
import torch
import torch.optim as optim
from torchvision import datasets
import random
import math
import pickle

from Utils import *
from Model import EFTransformer


class GroupDataset(torch.utils.data.Dataset):
    def __init__(self, args, phase ):
        assert phase == 'train' or phase == 'test', "Phase must be train or test."
        assert args.dataset == 'Tennis' or args.dataset == 'Dance', "Dataset must be Tennis or Dance."
        if phase == 'train':
            f = open("{}/{}TrainingSet.cpkl".format(args.dataDir,args.dataset), 'rb')
            self.feat = pickle.load(f)
            f.close()
        else:
            f = open("{}/{}TestingSet.cpkl".format(args.dataDir,args.dataset), 'rb')
            self.feat = pickle.load(f)
            f.close()

        self.phase = phase
        self.dataset  = args.dataset

        self.batch_size = args.batchSize
        self.nSamples = self.feat[0].size(0)

    def __len__(self):
        return self.nSamples
        
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        sFeat = self.feat[0][index]
        gFeat = self.feat[1][index]
        if self.phase == 'train' and self.dataset == 'Tennis':
            gFeat = self.RandomFlipping(gFeat)
        return (sFeat,gFeat)

    def RandomFlipping(self, gt):
        hFlip = random.randint(0,1)
        if hFlip == 1:
            gt[:,:,0] = 1 - gt[:,:,0]
            gt[:,:,2] = 0 - gt[:,:,2]
            gt[:,:,4] = 0 - gt[:,:,4]
        return gt

def Train(args):
    if not os.path.exists(args.weightDir):
        os.makedirs(args.weightDir)

    seed = int(time.time())
    torch.manual_seed(seed)
    if args.useCUDA:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        torch.cuda.manual_seed(seed)
   
    if args.initEpoch > 0:
        weightfile = "{}/EFTransformer_{}_Ep_{}".format(args.weightDir,args.dataset,args.initEpoch)
        print 'Loading model from {}'.format(weightfile)
        model = torch.load(weightfile)
    else:
        model = EFTransformer(args)

    if args.useCUDA:
        model = model.cuda()  

    optimizer = optim.Adam(model.parameters(), lr=args.learningRate/args.batchSize)

    trainDataset = GroupDataset(args, args.phase)
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batchSize, shuffle=True)

    testDataset = GroupDataset(args, 'test')
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=args.batchSize, shuffle=False)
    model.seen = args.initEpoch * len(trainDataset)

    gGt = testLoader.dataset.feat[1][:,args.tarPos,1:,2:4]
    sGt = testLoader.dataset.feat[0][:,args.tarPos,1:,:]

    for epoch in range(args.initEpoch, args.maxEpochs): 
        print 'Training Epoch %d, Training Set Size = %d' % (epoch+1,len(trainDataset))
        TrainEpoch(args, model, optimizer, trainLoader)
        if ( epoch + 1 ) % args.saveInterval == 0:
            torch.save(model, "{}/EFTransformer_{}_Ep_{}".format(args.weightDir,args.dataset,epoch+1))

            print "Testing Epoch %d, Testing Set Size = %d" % (epoch+1,len(testDataset))
            gResults, sResults = TestEpoch(args, model, testLoader)
            f = open("{}/EFTransformer_{}_Ep_{}.res".format(args.resultDir,args.dataset,epoch+1), "wb")
            pickle.dump((gResults, sResults), f, protocol=2)
            f.close()

            mad, fad = EvalEpochTrajectory(args, gResults[:,:,:,2:4], gGt)
            confusionMat, f1 = EvalEpochAction(args, sResults, sGt)
            
            print "Epoch %d: MAD = %0.2f, FAD = %0.2f, MacroF1 = %0.2f." % ( epoch+1, mad[-1], fad[-1], f1 )

    Eval(args)

def Test(args):
    if not os.path.exists(args.resultDir):
        os.makedirs(args.resultDir)

    seed = int(time.time())
    torch.manual_seed(seed)
    if args.useCUDA:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        torch.cuda.manual_seed(seed)

    testDataset = GroupDataset(args,args.phase)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=args.batchSize, shuffle=False)

    for epoch in range(args.maxEpochs): 
        if ( epoch + 1 ) % args.saveInterval != 0:
            continue

        print 'Testing Epoch %d, Testing Set Size = %d' % (epoch+1,len(testDataset))
        weightfile = "{}/EFTransformer_{}_Ep_{}".format(args.weightDir,args.dataset,epoch+1)
        print "    Loading model from {}".format(weightfile)
        model = torch.load(weightfile)
        gResults, sResults = TestEpoch(args, model, testLoader)

        f = open("{}/EFTransformer_{}_Ep_{}.res".format(args.resultDir,args.dataset,epoch+1), "wb")
        pickle.dump((gResults, sResults), f, protocol=2)
        f.close()

def TestExample(args):
    if not os.path.exists(args.resultDir):
        os.makedirs(args.resultDir)

    seed = int(time.time())
    torch.manual_seed(seed)
    if args.useCUDA:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        torch.cuda.manual_seed(seed)

    testDataset = GroupDataset(args,args.phase)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=args.batchSize, shuffle=False)    

    print 'Testing Example, Testing Set Size = %d' % len(testDataset)
    print "    Loading model from {}".format(args.weightFile)
    model = torch.load(args.weightFile)
    gResults, sResults = TestEpoch(args, model, testLoader)

    f = open(args.weightFile.replace('weights','results'), "wb")
    pickle.dump((gResults, sResults), f, protocol=2)
    f.close()

    gGt = testDataset.feat[1][:,args.tarPos,1:,2:4]
    sGt = testDataset.feat[0][:,args.tarPos,1:,:]
    oriDimS = args.oriDimS - 1 if args.dataset == 'Tennis' else args.oriDimS 
    evalRes = torch.zeros(1,9+oriDimS*oriDimS)
    mad, fad = EvalEpochTrajectory(args, gResults[:,:,:,2:4], gGt)
    confusionMat, f1 = EvalEpochAction(args, sResults, sGt)
    evalRes[0,0:4] = mad
    evalRes[0,4:8] = fad
    evalRes[0,8] = f1
    evalRes[0,9:] = confusionMat.view(-1)
    print "MAD = %0.2f, FAD = %0.2f, MacroF1 = %0.2f." % ( mad[-1], fad[-1], f1 )
    np.savetxt('Evaluation.txt',evalRes.numpy(),fmt='%0.2f')

def Eval(args):
    f = open("{}/{}TestingSet.cpkl".format(args.dataDir,args.dataset), 'rb')
    testingData = pickle.load(f)
    f.close()

    gGt = testingData[1][:,args.tarPos,1:,2:4]
    sGt = testingData[0][:,args.tarPos,1:,:]

    oriDimS = args.oriDimS - 1 if args.dataset == 'Tennis' else args.oriDimS 
    evalRes = torch.zeros(args.maxEpochs/args.saveInterval,9+oriDimS*oriDimS)
    for epoch in range(args.maxEpochs): 
        if ( epoch + 1 ) % args.saveInterval != 0:
            continue

        f = open("{}/EFTransformer_{}_Ep_{}.res".format(args.resultDir,args.dataset,epoch+1), "rb")
        gResults, sResults = pickle.load(f)
        f.close()

        mad, fad = EvalEpochTrajectory(args, gResults[:,:,:,2:4], gGt)
        confusionMat, f1 = EvalEpochAction(args, sResults, sGt)
        evalRes[(epoch+1)/args.saveInterval-1,0:4] = mad
        evalRes[(epoch+1)/args.saveInterval-1,4:8] = fad
        evalRes[(epoch+1)/args.saveInterval-1,8] = f1
        evalRes[(epoch+1)/args.saveInterval-1,9:] = confusionMat.view(-1)

        print "Epoch %d: MAD = %0.2f, FAD = %0.2f, MacroF1 = %0.2f." % ( epoch+1, mad[-1], fad[-1], f1 )

    np.savetxt('Evaluation.txt',evalRes.numpy(),fmt='%0.2f')

def TrainEpoch(args, model, optimizer, trainLoader):
    model.train()

    for batch_idx, (sFeat, gFeat) in enumerate(trainLoader):        
        (nB,nP,nF,nDS) = sFeat.data.size()
        (nB,nP,nF,nDG) = gFeat.data.size()
        model.seen = model.seen + nB

        sFeatOb, sFeatTa, gFeatOb, gFeatTa = ObvTarSplit(args, sFeat, gFeat, 'train')

        if args.useCUDA:
            sFeatOb, sFeatTa, gFeatOb, gFeatTa = sFeatOb.cuda(), sFeatTa.cuda(), gFeatOb.cuda(), gFeatTa.cuda()

        optimizer.zero_grad()

        gPred, sPred, gLoss, sLoss, gStat, sStat = model(sFeatOb, sFeatTa, gFeatOb, gFeatTa)
        loss = gLoss + sLoss * 0.1 
        loss.backward()
        optimizer.step()

        gs = gStat
        ss = sStat 
        print '%6d: nMiss %3d,%3d,%3d, FrameError: %0.3f,%0.3f,%0.3f, Pos:%0.3f, Neg:%0.3f, LossG=%0.3f, LossS=%0.3f, loss=%0.3f' % \
            (model.seen,gs[0][0],gs[0][1],gs[0][2],gs[1][2],gs[1][5],gs[1][8],ss[0],ss[1], gLoss.data[0], sLoss.data[0], loss.data[0])

        fid = open( './TrainingLog.txt', 'a' )
        fid.write( str(model.seen)+" "+str(gs[0][0])+" "+str(gs[0][1])+" "+str(gs[0][2])+" " )
        fid.write( "%0.3f %0.3f"%(ss[0], ss[1]) + " %0.3f %0.3f %0.3f\n"%(gLoss.data[0], sLoss.data[0],loss.data[0]) )
        fid.close()


def TestEpoch(args, model, testLoader):
    model.eval()
    nClip = len(testLoader.dataset)
    nPta  = len(args.tarPos)
    nFP   = args.clipLength - 1
    nDGP  = args.dimPred
    nDSP  = args.oriDimS

    gResults = torch.zeros(nClip,nPta,nFP,nDGP)
    sResults = torch.zeros(nClip,nPta,nFP,nDSP)
    clipIdx  = 0
    for batch_idx, (sFeat, gFeat) in enumerate(testLoader):
        (nB,nP,nF,nDS) = sFeat.data.size()
        (nB,nP,nF,nDG) = gFeat.data.size()

        sFeatOb, sFeatTa, gFeatOb, gFeatTa = ObvTarSplit(args, sFeat, gFeat, 'test')

        if args.useCUDA:
            sFeatOb, sFeatTa, gFeatOb, gFeatTa = sFeatOb.cuda(), sFeatTa.cuda(), gFeatOb.cuda(), gFeatTa.cuda()

        gPred, sPred, gLoss, sLoss, gStat, sStat = model(sFeatOb, sFeatTa, gFeatOb, gFeatTa)

        gResults[clipIdx:clipIdx+nB] = Convert2CPU(gPred.data).view(nB,nPta,nFP,nDGP)
        if args.predFlag[1] == 1:
            sResults[clipIdx:clipIdx+nB] = Convert2CPU(sPred.data).view(nB,nPta,nFP,nDSP)
        clipIdx = clipIdx + nB

    return gResults, sResults

def EvalEpochTrajectory(args, res, gt):
    (nC,nP,nF,nD) = gt.size()

    gtLengthIdx, gtLengthCount = LengthClassify(gt, args.dataset)
    width  = 1280 if args.dataset == 'Tennis' else 640
    height = 720 if args.dataset == 'Tennis' else 480

    dist = res - gt
    dist = torch.sqrt( dist[:,:,:,0]*dist[:,:,:,0]*width*width + dist[:,:,:,1]*dist[:,:,:,1]*height*height ) # nCxnPxnF

    madAll = dist.sum(2) / nF
    fadAll = dist[:,:,-1]

    mad = torch.zeros(4)
    fad = torch.zeros(4)

    for cId in range(nC):
        for pId in range(nP):
            mad[int(gtLengthIdx[cId,pId])] += madAll[cId,pId]
            mad[-1] += madAll[cId,pId]

            fad[int(gtLengthIdx[cId,pId])] += fadAll[cId,pId]
            fad[-1] += fadAll[cId,pId]

    for l in range(3):
        mad[l] /= gtLengthCount[l]
        fad[l] /= gtLengthCount[l]
    mad[-1] /= (nC*nP)
    fad[-1] /= (nC*nP)

    return mad, fad

def EvalEpochAction(args, res, gt):
    (nC,nP,nF,nD) = gt.size()
    oriDimS = args.oriDimS - 1 if args.dataset == 'Tennis' else args.oriDimS 

    confusionMat = torch.zeros(oriDimS,oriDimS)
    if args.predFlag[1] == 0:
        return confusionMat, 0

    res = torch.argmax(res,3)
    gt  = torch.argmax(gt,3)
    for cId in range(nC):
        for pId in range(nP):
            for fId in range(nF):
                confusionMat[gt[cId,pId,fId],res[cId,pId,fId]] += 1

    f1 = macroF1(confusionMat)

    for action in range(oriDimS):
        confusionMat[action,:] /= confusionMat[action,:].sum()

    return confusionMat, f1






