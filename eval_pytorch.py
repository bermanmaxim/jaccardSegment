from __future__ import print_function, division

import argparse
from datetime import datetime
import os, sys
from os.path import join
import time
import re
import platform

import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as F

import random
# WARNING: if multiple worker threads, the seeds are useless.
random.seed(1857)
torch.manual_seed(1857)
torch.cuda.manual_seed(1857)

from settings import get_arguments
import datasets
from datasets.loadvoc import load_extended_voc
from compose import (JointCompose, RandomScale, Normalize,
                     RandomHorizontalFlip, RandomCropPad, PILtoTensor, Scale, TensortoPIL)
from PIL.Image import NEAREST

from losses import *

import deepdish as dd
import deeplab_resnet.model_pytorch as modelpy
from collections import defaultdict
import yaml

IGNORE_LABEL = 255
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def create_variables(weights, cuda=True):
    var = dict()
    for k, v in weights.items():
        v = torch.from_numpy(v)
        if cuda:
            v = v.cuda()
        if not (k.endswith('moving_mean') or k.endswith('moving_variance')):
            v = Variable(v)
        var[k] = v
    return var

def snapshot_variables(weights, dest):
    out = {}
    for (k, v) in weights.items():
        if isinstance(v, Variable):
            v = v.data
        out[k] = v.cpu().numpy()
    dd.io.save(dest, out)

def training_groups(weights, base_lr, multipliers=[0.1, 1.0, 1.0], train_last=-1, hybrid=False): # multipliers=[1.0, 10.0, 20.0]
    """
    get training groups and activates requires_grad for variables
    train_last: last: only train last ... layers
    hybrid: if hybrid, train all layers but set momentum to 0 on last layers
    """
    fixed = ['moving_mean', 'moving_variance', 'beta', 'gamma']
    # get training variables, with their lr
    trained = {k: v for (k, v) in weights.iteritems() if not any([k.endswith(s) for s in fixed])}
    for v in trained.values():
        v.requires_grad = True
    fc_vars = {k: v for (k, v) in trained.iteritems() if 'fc' in k}
    conv_vars = [v for (k, v) in trained.items() if 'fc' not in k] # lr * 1.0
    fc_w_vars = [v for (k, v) in fc_vars.items() if 'weights' in k] # lr * 10.0
    fc_b_vars = [v for (k, v) in fc_vars.items() if 'biases' in k]  # lr * 20.0
    assert(len(trained) == len(fc_vars) + len(conv_vars))
    assert(len(fc_vars) == len(fc_w_vars) + len(fc_b_vars))
    if train_last == -1:
        print("train all layers")
        groups = [{'params': conv_vars, 'lr': multipliers[0] * base_lr},
                  {'params': fc_w_vars, 'lr': multipliers[1] * base_lr},
                  {'params': fc_b_vars, 'lr': multipliers[2] * base_lr}]
    elif train_last == 1:
        print("train last layer only")
        for v in conv_vars:
            v.requires_grad = False
        groups = [{'params': fc_w_vars, 'lr': multipliers[1] * base_lr},
                  {'params': fc_b_vars, 'lr': multipliers[2] * base_lr}]
    return groups

class SegsetWrap(data.Dataset):
    def __init__(self, segset, transform=None):
        self.name = segset.name
        self.segset = segset
        self.transform = transform
    def __repr__(self):
        return "<SegSetWrap '" + self.name + "'>"
    def __getitem__(self, i):
        inputs = self.segset.read(i, kind="PIL")
        if self.transform is not None:
            inputs = self.transform(inputs)
        return inputs
    def __len__(self):
        return len(self.segset)

def main(args):
    
    print(os.path.basename(__file__), 'arguments:')
    print(yaml.dump(vars(args), default_flow_style=False))

    weights = dd.io.load(args.restore_from)
    print('Loaded weights from {}'.format(args.restore_from))
    weights = create_variables(weights, cuda=True)
    forward = lambda input: modelpy.DeepLabResNetModel({'data': input}, weights).layers['fc1_voc12']
    train, val, test = load_extended_voc()
    input_size = map(int, args.input_size.split(',')) if args.input_size is not None else None
    print ('========')

    if args.proximal:
        assert args.jaccard

    if args.binary == -1:
        print("Multiclass: loss set to cross-entropy")
        lossfn, lossname = crossentropyloss, 'xloss'
        otherlossfn = None
    else:
        print("Binary: loss set to hingeloss")
        if args.jaccard:
            lossfn, lossname = lovaszloss, 'lovaszloss'
            otherlossfn, otherlossname = hingeloss, 'hingeloss'
        elif args.softmax:
            lossfn, lossname = binaryXloss, 'binxloss'
            otherlossfn = None
        else:
            lossfn, lossname = hingeloss, 'hingeloss'
            otherlossfn, otherlossname = lovaszloss, 'lovaszloss'
        train, val = train.binarize(args.binary_str), val.binarize(args.binary_str)


    # get network output size
    dummy_input = torch.rand((1, 3, input_size[0], input_size[1])).cuda()
    dummy_out = forward(Variable(dummy_input, volatile=True))
    output_size = (dummy_out.size(2), dummy_out.size(3))

    transforms_val = JointCompose([PILtoTensor(), 
                                   [Normalize(torch.from_numpy(IMG_MEAN)), None],
                                  ])
    invtransf_val = JointCompose([[Normalize(-torch.from_numpy(IMG_MEAN)), None],
                                   TensortoPIL( datasets.utils.color_map() ), 
                                 ])

    if args.sampling == 'balanced':
        from datasets.balanced_val import balanced
        inds = balanced[args.binary_str]
        val.examples = [val[i] for i in inds]
        print('Subsampled val. to balanced set of {:d} examples'.format(len(val)))
    elif args.sampling == 'exclusive':
        val = val[args.binary_str]
        print('Subsampled val. to balanced set of {:d} examples'.format(len(val)))

    update_every = args.grad_update_every
    global_batch_size = args.batch_size * update_every

    valset = SegsetWrap(val, transforms_val)
    valloader = data.DataLoader(valset, 
                         batch_size=1, 
                         shuffle=False,
                         num_workers=1, 
                         pin_memory=True)

    def do_val():
        valiter = iter(valloader)
        stats = defaultdict(list)
        # extract some images spreak evenly in the validation set
        tosee = [int(0.05 * i * len(valiter)) for i in range(1, 20)]
        for valstep, (inputs, labels) in enumerate(valiter):
            start_time = time.time()
            inputs, labels = Variable(inputs.cuda(), volatile=True), labels.cuda().long()
            logits = forward(inputs)
            logits = F.upsample_bilinear(logits, size=labels.size()[1:])
            if args.binary == -1:
                xloss = crossentropyloss(logits, labels)
                stats['xloss'].append(xloss.data[0])
                print('[Validation {}-{:d}], xloss {:.5f} - mean {:.5f}  ({:.3f} sec/step {})'.format(
                         step, valstep, xloss, np.mean(stats['xloss']), time.time() - start_time))
                # conf, pred = logits.max(1)
            else:
                conf, multipred = logits.max(1)
                multipred = multipred.squeeze(1)
                multipred = (multipred == args.binary).long()
                imageiou_multi = iouloss(multipred.data.squeeze(0), labels.squeeze(0))
                stats['imageiou_multi'].append(imageiou_multi)

                logits = logits[:, args.binary, :, :]   # select only 1 output
                pred = (logits > 0.).long()

                # image output
                if valstep in tosee:
                    inputim, inputlab = invtransf_val([inputs.data[0, :, :, :], labels[0, :, :]])
                    _, predim = invtransf_val([inputs.data[0, :, :, :], pred.data[0, :, :]])
                    inputim.save("imout/{}_{}in.png".format(args.nickname, valstep),"PNG")
                    inputlab.save("imout/{}_{}inlab.png".format(args.nickname, valstep),"PNG")
                    predim.save("imout/{}_{}out.png".format(args.nickname, valstep),"PNG")

                imageiou = iouloss(pred.data.squeeze(0), labels.squeeze(0))
                stats['imageiou'].append(imageiou)
                hloss = hingeloss(logits, labels).data[0]
                stats['hingeloss'].append(hloss)
                jloss = lovaszloss(logits, labels).data[0]
                stats['lovaszloss'].append(jloss)
                binxloss = binaryXloss(logits, labels).data[0]
                stats['binxloss'].append(binxloss)

                print(   'hloss {:.5f} - mean {:.5f}, '.format(hloss, np.mean(stats['hingeloss']))
                       + 'lovaszloss {:.5f} - mean {:.5f}, '.format(jloss, np.mean(stats['lovaszloss']))
                       + 'iou {:.5f} - mean {:.5f}, '.format(imageiou, np.mean(stats['imageiou']))
                       + 'iou_multi {:.5f} - mean {:.5f}, '.format(imageiou_multi, np.mean(stats['imageiou_multi']))
                     )

    do_val()



if __name__ == '__main__':
    args = get_arguments(sys.argv[1:], 'train')
    main(args)