# EF-Transformer
This repository is the code for Entry-Flipped Transformer (EF-Transformer) for Inference and Prediction of Participant Behavior. The task is to estimate the behavior (trajectory and action labels) of target participants of a group based on information of some observed participants of the same group.

## Environment
The code was tested on Ubuntu 18.04, with pytorch 0.4.1, torchvision 0.2.1, numpy 1.14.3, and python 2.7.

## Datasets
- ```Tennis dataset```: A dataset of tennis double videos. There are 5 participants (4 players and the ball), 3170 clips for training and 1735 clips for testing.
- ```Dance dataset```: A dataset of [Edinburgh Ceilidh](https://homepages.inf.ed.ac.uk/rbf/CEILIDHDATA/) dance videos. There are 10 participants, 2758 clips for training and 96 clips for testing.

## Training
The training mode includes training, testing, and evaluation:
```
python Train.py 
```
- Set ```--dataset``` from 'Tennis' and 'Dance' for training and testing data.
- Set ```--tarPos``` from [0,1] for tennis dataset or from [0,9] for dance dataset, *e.g.* ```--tarPos 0,1```.
- Set ```--action``` from 0 or 1 which indicates estimate action labels or not.
- Set ```--future``` form [0,10] for reachable future frames during training and testing.

**Default** mode is ```--dataset Tennis --tarPos 0 --action 1 --future 0```. 
## Testing
If you want to test the model with existing weights, run 
```
python Train.py --phase test
```
with the same setting as training. An example weight file is provided for the default mode, to test and evaluate the example, run 
```
python Train.py --phase test --weightFile weights/EFTransformer_Tennis_Example
```

## Evaluation
Mean average displacement (MAD)I, Final average displacement (FAD), confusion matrix, and Macro F1-score are provided to evaluate the model. If you want to evaluate the model with existing results, run 
```
python Train.py --phase eval
```
with the same setting as training. Output of evaluation is saved in *Evaluation.txt*, where the columns [0,3] are MAD, columns [4,8] are FAD, column 9 is Macro F1-score, and the rest is the flattened confusion matrix.

