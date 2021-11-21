import os
import time
import torch
import argparse

from src.Processor import Train, Test, Eval, TestExample

def get_parser():
    parser = argparse.ArgumentParser(description='EF-Transformer')
    parser.add_argument('--phase', default='train', help='Set this value to train, test, or eval')
    parser.add_argument('--dataset', default='Tennis', help='Set this value to Tennis or Dance')
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--useCUDA',default=1, type=int)
    parser.add_argument('--saveInterval',default=10, type=int)
    parser.add_argument('--weightDir', default='weights')
    parser.add_argument('--resultDir', default='results')
    parser.add_argument('--dataDir', default='data')
    parser.add_argument('--weightFile', default='None', help='Set this value if you want to test with one existing weights file.')
    # Net Info
    parser.add_argument('--learningRate', default=0.001, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--maxEpochs', default=1000, type=int)
    parser.add_argument('--initEpoch', default=0, type=int)
    parser.add_argument('--batchSize', default=20, type=int)
    parser.add_argument('--tarPos', default='0', 
                        help='Position of target participants, choose from [0,1] for tennis dataset and [0,9] for dance dataset')
    parser.add_argument('--action', default=1, type=int,  
                        help='indicate estimate action or not')
    # EF-Transformer
    parser.add_argument('--nHead', default=8, type=int)
    parser.add_argument('--nLayer', default=2, type=int)
    parser.add_argument('--future', default=0, type=int, help='Reachable future frames')
    parser.add_argument('--clipLength', default=10, type=int)
    parser.add_argument('--inChannelsQ', default=128, type=int)
    parser.add_argument('--inChannelsK', default=128, type=int)
    parser.add_argument('--inChannelsV', default=128, type=int)
    parser.add_argument('--interChannelsQK', default=64, type=int)
    parser.add_argument('--interChannelsV', default=64, type=int)
    parser.add_argument('--ffnChannels', default=64, type=int)
    parser.add_argument('--dimS', default=64, type=int)
    parser.add_argument('--dimG', default=64, type=int)
    parser.add_argument('--dimPred', default=6, type=int)
    parser.add_argument('--oriDimG', default=7, type=int)
    parser.add_argument('--oriDimS', default=8, type=int,
                        help='number of action categories, 8 for tennis dataset and 6 for dance dataset')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args.tarPos = [int(i) for i in args.tarPos.split(',')]
    args.oriDimS = 8 if args.dataset == 'Tennis' else 6
    args.predFlag = [1,args.action]

    if args.phase == 'train':
        Train(args)

    if args.phase == 'test':
        if args.weightFile != 'None':
            TestExample(args)
        else:
            Test(args)

    if args.phase == 'eval':
        Eval(args)
