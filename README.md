# Deeplab-resnet-101 Pytorch with Jaccard loss

Train deeplab-resnet-101 with binary Jaccard loss surrogate, the Jaccard hinge.

Parts of the code is adapted from [tensorflow-deeplab-resnet](https://github.com/DrSleep/) (in particular the conversion from caffe to tensorflow with kaffe).

The code has not been tested for full training of Deeplab-Resnet yet. Refer to [tensorflow-deeplab-resnet](https://github.com/DrSleep/tensorflow-deeplab-resnet) and possibly extract the weights after training with that framework.

## Code status
The code is in early stage. Pull requests welcome.

## Dependencies and weights
Relies notably on [Pytorch](http://pytorch.org/) and the standalone [tensorboard](https://github.com/dmlc/tensorboard/tree/master/python) package

Using anaconda, install the full requirements using the provided conda environment file:
```
conda env create --f environemnt.yml
source activate jaccard-segment
```

Convert the Deeplab Caffe weights to tensorflow ckpt using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow), then convert them to hdf5 using `ckpt_to_dd.py` and use our wrapper to load in Pytorch.

## Important switches in the settings
By default, finetunes with cross-entropy loss. Use --binary `class` switch for selecting a particular class in the binary case, `--jaccard` for training with the Jaccard hinge loss described in the arxiv paper, `--hinge` to use the Hinge loss, and `--proximal` to use the prox. operator optimization variant for the Jaccard loss as described in the arxiv paper.

For the prox. operator, use a learning rate of `1.` and set an equivalent regularization of `1/lr` instead.
