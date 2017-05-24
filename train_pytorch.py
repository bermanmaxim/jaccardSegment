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
# WARNING: if multiple worker threads, the seeds are useless (no warranty on the execution order)
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

def training_groups(weights, base_lr, multipliers=[0.1, 1.0, 1.0], train_last=-1):
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
        print("train last layer")
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
    else:
        if args.jaccard:
            print("loss set to jaccard hinge")
            lossfn, lossname = lovaszloss, 'lovaszloss'
        elif args.hinge:
            print("loss set to hinge loss")
            lossfn, lossname = hingeloss, 'hingeloss'
        else:
            print("loss set to binary cross-entropy")
            lossfn, lossname = binaryXloss, 'binxloss'
        train, val = train.binarize(args.binary_str), val.binarize(args.binary_str)

    # get network output size
    def get_size():
        dummy_input = torch.rand((1, 3, input_size[0], input_size[1])).cuda()
        dummy_out = forward(Variable(dummy_input, volatile=True))
        output_size = (dummy_out.size(2), dummy_out.size(3))
        return output_size
    output_size = get_size()

    base_lr = args.learning_rate
    groups = training_groups(weights, base_lr, train_last=args.train_last, hybrid=args.hybrid)
    optimizer = optim.SGD(groups, lr=base_lr, momentum=args.momentum)
    groups_lr = [group['lr'] for group in optimizer.param_groups]

    transforms_train = JointCompose([RandomScale(0.5, 1.5) if args.random_scale else None, 
                                     RandomHorizontalFlip() if args.random_mirror else None,
                                     RandomCropPad(input_size, (0, IGNORE_LABEL)),
                                     [None, Scale((output_size[1], output_size[0]), NEAREST)],
                                     PILtoTensor(), 
                                     [Normalize(torch.from_numpy(IMG_MEAN)), None],
                                    ])
    transforms_val = JointCompose([PILtoTensor(), 
                                   [Normalize(torch.from_numpy(IMG_MEAN)), None],
                                  ])
    invtransf_val = JointCompose([[Normalize(-torch.from_numpy(IMG_MEAN)), None],
                                   TensortoPIL( datasets.utils.color_map() ), 
                                 ])

    if args.sampling == 'sequential':
        trainset = SegsetWrap(train, transforms_train)
        sampler = data.sampler.SequentialSampler(trainset)
    elif args.sampling == 'shuffle':
        trainset = SegsetWrap(train, transforms_train)
        sampler = data.sampler.RandomSampler(trainset)
    elif args.sampling == 'balanced':
        trainset = SegsetWrap(train, transforms_train)
        positives = np.array([(args.binary_str in ex.classes) for ex in train])
        sample_weights = np.zeros(len(positives))
        sample_weights[positives] = 0.5 / positives.sum()
        sample_weights[~positives] = 0.5 / (~positives).sum()
        sampler = data.sampler.WeightedRandomSampler(sample_weights, len(train))
        from datasets.balanced_val import balanced
        inds = balanced[args.binary_str]
        val.examples = [val[i] for i in inds]
        print('Subsampled val. to balanced set of {:d} examples'.format(len(val)))
    elif args.sampling == 'exclusive':
        train, val = train[args.binary_str], val[args.binary_str]
        trainset = SegsetWrap(train, transforms_train)
        sampler = data.sampler.RandomSampler(trainset)
        print('Subsampled train, val. to balanced set of {}, {} examples'.format(len(train), len(val)))

    update_every = args.grad_update_every
    global_batch_size = args.batch_size * update_every

    trainloader = data.DataLoader(trainset, 
                         batch_size=global_batch_size, 
                         sampler=sampler, 
                         num_workers=args.threads, 
                         pin_memory=True)

    valset = SegsetWrap(val, transforms_val)
    valloader = data.DataLoader(valset, 
                         batch_size=1, 
                         shuffle=False,
                         num_workers=1, 
                         pin_memory=True)

    step = args.start_step
    finished = False
    epoch = 0

    from tensorboard import SummaryWriter
    logdir = join(args.expname + '_logs', args.nickname)
    if os.path.exists(logdir):
        if args.delete_previous:
            var = 'y'
        else:
            var = raw_input(logdir + " already exists. Delete (y/n)? ")
        if var == 'n':
            raise ValueError(logdir + " already exists")
        elif var == 'y':
            import shutil
            shutil.rmtree(logdir)
    log_writer = SummaryWriter(logdir)
#    train_writer = SummaryWriter(log_train)

    def snapshot():
        dest = join(args.snapshot_dir, '{}-{}-{:02d}.h5'.format(args.expname, args.nickname, step))
        snapshot_variables(weights, dest)
        print("[{}] step {:d}: saved weights under {}".format(dt, step, dest))

    def do_val():
        valiter = iter(valloader)
        stats = defaultdict(list)
        tosee = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # export some outputs images of the validation set
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
                    log_writer.add_image(str(valstep)+'im', np.array(inputim.convert("RGB")))
                    log_writer.add_image(str(valstep)+'lab', np.array(inputlab.convert("RGB")))
                    log_writer.add_image(str(valstep)+'pred', np.array(predim.convert("RGB")))

                imageiou = iouloss(pred.data.squeeze(0), labels.squeeze(0))
                stats['imageiou'].append(imageiou)
                hloss = hingeloss(logits, labels).data[0]
                stats['hingeloss'].append(hloss)
                jloss = lovaszloss(logits, labels).data[0]
                stats['lovaszloss'].append(jloss)
                binxloss = binaryXloss(logits, labels).data[0]
                stats['binxloss'].append(binxloss)

                print(   '[Validation {}-{:d}], '.format(step, valstep)
                       + 'hloss {:.5f} - mean {:.5f}, '.format(hloss, np.mean(stats['hingeloss']))
                       + 'lovaszloss {:.5f} - mean {:.5f}, '.format(jloss, np.mean(stats['lovaszloss']))
                       + 'iou {:.5f} - mean {:.5f}, '.format(imageiou, np.mean(stats['imageiou']))
                       + 'iou_multi {:.5f} - mean {:.5f}, '.format(imageiou_multi, np.mean(stats['imageiou_multi']))
                       + '({:.3f} sec/step)'.format(time.time() - start_time)
                     )
        for key in stats:
            log_writer.add_scalar(key + '_val', np.mean(stats[key]), step)

    if not args.no_startval:
        do_val()

    num_steps = args.num_steps
    if args.epochs:
        num_steps *= len(trainloader)
    num_steps = int(num_steps)
    if args.new_schedule:
        half_step = num_steps // 2

    while not finished: # new epoch
        trainiter = iter(trainloader)
        def train_step():
            if args.new_schedule and step == half_step:
                print("==== HALF STEP ====")
                for group, group_base in zip(optimizer.param_groups, groups_lr):
                    if ('fix_lr' not in group) or not group['fix_lr']:
                        group['lr'] = group_base / 5

            inputs, labels = next(trainiter)
            inputs, labels = Variable(inputs.cuda()), labels.cuda().long()
            chunk_inp = torch.split(inputs, args.batch_size, dim=0)
            chunk_lab = torch.split(labels, args.batch_size, dim=0)
            optimizer.zero_grad()
            lossacc = 0.
            # Start gradient accumulation
            for inp, lab in zip(chunk_inp, chunk_lab):
                logits = forward(inp)
                if args.binary != -1:
                    logits = logits[:, args.binary, :, :]   # select only 1 output
                if args.proximal:
                    debug = {"step": -1, "finished": False}
                    proxreg = args.proxreg
                    if args.power_prox > 0:
                        proxreg = proxreg / (1. - step/(num_steps + 0.1)) ** args.power_prox
                    if args.new_schedule:
                        if step >= half_step:
                            proxreg *= 5.
                    loss, hook, gam = lossfn(logits, lab, prox=proxreg, max_steps=args.maxproxsteps, debug=debug)
                    print(str(debug["step"]) + ('' if debug["finished"] else 'E'), end=' ')
                else:
                    loss = lossfn(logits, lab)
                loss.backward( torch.Tensor([1. / len(chunk_inp)]).cuda() ) # rescale gradient
                if args.proximal:
                    hook.remove() # remove hook to free memory
                lossacc += loss.data[0] / len(chunk_inp)
            optimizer.step()
            return lossacc

        for substep in range(len(trainloader)):
            start_time = time.time()
            step += 1
            if step > num_steps:
                finished = True
                break
            lossacc = train_step()
            
            duration = time.time() - start_time
            (dt, micro) = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f').split('.')
            dt = "%s.%03d" % (dt, int(micro) / 1000)
            print('[{}] step {:d} \t loss = {:.5f} ({:.3f} sec/step, epoch {})'.format(
                         dt, step, lossacc, duration, epoch))

            log_writer.add_scalar(lossname, lossacc, step)

            if step % args.save_pred_every == 0:
                snapshot()
            if step % args.do_val_every == 0:
                do_val()

        epoch += 1
    # end of main: save weights and do val
    snapshot()
    do_val()



if __name__ == '__main__':
    args = get_arguments(sys.argv[1:], 'train')
    main(args)
