from __future__ import print_function, division
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

VOID_LABEL = 255
N_CLASSES = 21

def crossentropyloss(logits, label):
    mask = (label.view(-1) != VOID_LABEL)
    nonvoid = mask.long().sum()
    if nonvoid == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    # if nonvoid == mask.numel():
    #     # no void pixel, use builtin
    #     return F.cross_entropy(logits, Variable(label))
    target = label.view(-1)[mask]
    C = logits.size(1)
    logits = logits.permute(0, 2, 3, 1)  # B, H, W, C
    logits = logits.contiguous().view(-1, C)
    mask2d = mask.unsqueeze(1).expand(mask.size(0), C).contiguous().view(-1)
    logits = logits[mask2d].view(-1, C)
    loss = F.cross_entropy(logits, Variable(target))
    return loss

class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()

def binaryXloss(logits, label):
    mask = (label.view(-1) != VOID_LABEL)
    nonvoid = mask.long().sum()
    if nonvoid == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    # if nonvoid == mask.numel():
    #     # no void pixel, use builtin
    #     return F.cross_entropy(logits, Variable(label))
    target = label.contiguous().view(-1)[mask]
    logits = logits.contiguous().view(-1)[mask]
    # loss = F.binary_cross_entropy(logits, Variable(target.float()))
    loss = StableBCELoss()(logits, Variable(target.float()))
    return loss

def naive_single(logit, label):
    # single images
    mask = (label.view(-1) != 255)
    num_preds = mask.long().sum()
    if num_preds == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    target = Variable(label.contiguous().view(-1)[mask].float())
    logit = logit.contiguous().view(-1)[mask]
    prob = F.sigmoid(logit)
    intersect = target * prob
    union = target + prob - intersect
    loss = (1. - intersect / union).sum()
    return loss

def hingeloss(logits, label):
    mask = (label.view(-1) != 255)
    num_preds = mask.long().sum()
    if num_preds == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    target = label.contiguous().view(-1)[mask]
    target = 2. * target.float() - 1.  # [target == 0] = -1
    logits = logits.contiguous().view(-1)[mask]
    hinge = 1./num_preds * F.relu(1. - logits * Variable(target)).sum()
    return hinge

def gamma_fast(gt, permutation):
    p = len(permutation)
    gt = gt.gather(0, permutation)
    gts = gt.sum()

    intersection = gts - gt.float().cumsum(0)
    union = gts + (1 - gt).float().cumsum(0)
    jaccard = 1. - intersection / union

    jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovaszloss(logits, labels, prox=False, max_steps=20, debug={}):
    # image-level Lovasz hinge
    if logits.size(0) == 1:
        # single image case
        loss = lovasz_single(logits.squeeze(0), labels.squeeze(0), prox, max_steps, debug)
    else:
        losses = []
        for logit, label in zip(logits, labels):
            loss = lovasz_single(logit, label, prox, max_steps, debug)
            losses.append(loss)
        loss = sum(losses) / len(losses)
    return loss

def naiveloss(logits, labels):
    # image-level Lovasz hinge
    if logits.size(0) == 1:
        # single image case
        loss = naive_single(logits.squeeze(0), labels.squeeze(0))
    else:
        losses = []
        for logit, label in zip(logits, labels):
            loss = naive_single(logit, label)
            losses.append(loss)
        loss = sum(losses) / len(losses)
    return loss

def iouloss(pred, gt):
    # works for one binary pred and associated target
    # make byte tensors
    pred = (pred == 1)
    mask = (gt != 255)
    gt = (gt == 1)
    union = (gt | pred)[mask].long().sum()
    if not union:
        return 0.
    else:
        intersection = (gt & pred)[mask].long().sum()
        return 1. - intersection / union

def compute_step_length(x, grad, active, eps=1e-6):
    # compute next intersection with an edge in the direction grad
    # OR next intersection with a 0 - border
    # returns: delta in ind such that:
    # after a step delta in the direction grad, x[ind] and x[ind+1] will be equal
    delta = np.inf
    ind = -1
    if active > 0:
        numerator = (x[:active] - x[1:active+1])           # always positive (because x is sorted)
        denominator = (grad[:active] - grad[1:active+1])
        # indices corresponding to negative denominator won't intersect
        # also, we are not interested in indices in x that are *already equal*
        valid = (denominator > eps) & (numerator > eps)
        valid_indices = valid.nonzero()
        intersection_times = numerator[valid] / denominator[valid]
        if intersection_times.size():        
            delta, ind = intersection_times.min(0)
            ind = valid_indices[ind]
            delta, ind = delta[0], ind[0, 0]
    if grad[active] > 0:
        intersect_zero = x[active] / grad[active]
        if intersect_zero > 0. and intersect_zero < delta:
            return intersect_zero, -1
    return delta, ind

def project(gam, active, members):
    tovisit = set(range(active + 1))
    while tovisit:
        v = tovisit.pop()
        if len(members[v]) > 1:
            avg = 0.
            for k in members[v]:
                if k != v: tovisit.remove(k)
                avg += gam[k] / len(members[v])
            for k in members[v]:
                gam[k] = avg
    if active + 1 < len(gam):
        gam[active + 1:] = 0.

def find_proximal(x0, gam, lam, eps=1e-6, max_steps=20, debug={}):
    # x0: sorted margins data
    # gam: initial gamma_fast(target, perm)
    # regularisation parameter lam
    x = x0.clone()
    act = (x >= eps).nonzero()
    finished = False
    if not act.size():
        finished = True
    else:
        active = act[-1, 0]
        members = {i: {i} for i in range(active + 1)}
        if active > 0:
            equal = (x[:active] - x[1:active+1]) < eps
            for i, e in enumerate(equal):
                if e:
                    members[i].update(members[i + 1])
                    members[i + 1] = members[i]
            project(gam, active, members)
    step = 0
    while not finished and step < max_steps and active > -1:
        step += 1
        res = compute_step_length(x, gam, active, eps)
        delta, ind = res
        
        if ind == -1:
            active = active - len(members[active])
        
        stop = torch.dot(x - x0, gam) / torch.dot(gam, gam) + 1. / lam
        if 0 <= stop < delta:
            delta = stop
            finished = True
        
        x = x - delta * gam
        if not finished:
            if ind >= 0:
                repr = min(members[ind])
                members[repr].update(members[ind + 1])
                for m in members[ind]:
                    if m != repr:
                        members[m] = members[repr]
            project(gam, active, members)
        if "path" in debug:
            debug["path"].append(x.numpy())

    if "step" in debug:
        debug["step"] = step
    if "finished" in debug:
        debug["finished"] = finished
    return x, gam


def lovasz_binary(margins, label, prox=False, max_steps=20, debug={}):
    # 1d vector inputs
    # Workaround: can't sort Variable bug
    # prox: False or lambda regularization value
    _, perm = torch.sort(margins.data, dim=0, descending=True)
    margins_sorted = margins[perm]
    grad = gamma_fast(label, perm)
    loss = torch.dot(F.relu(margins_sorted), Variable(grad))
    if prox is not False:
        xp, gam = find_proximal(margins_sorted.data, grad, prox, max_steps=max_steps, eps=1e-6, debug=debug)
        hook = margins_sorted.register_hook(lambda grad: Variable(margins_sorted.data - xp))
        return loss, hook, gam
    else:
        return loss


def lovasz_single(logit, label, prox=False, max_steps=20, debug={}):
    # single images
    mask = (label.view(-1) != 255)
    num_preds = mask.long().sum()
    if num_preds == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    target = label.contiguous().view(-1)[mask]
    signs = 2. * target.float() - 1.
    logit = logit.contiguous().view(-1)[mask]
    margins = (1. - logit * Variable(signs))
    loss = lovasz_binary(margins, target, prox, max_steps, debug=debug)
    return loss


