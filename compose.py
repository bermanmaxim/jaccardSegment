"""
https://github.com/pytorch/vision/issues/9

joint transforms for input and target
applies to sequences of images

transform = JointCompose([
    ElasticTransform(),
    RandomRotate(),
    [CenterCropNumpy(size=input_shape), CenterCropNumpy(size=target_shape)],
    [NormalizeNumpy(), None],
    [Lambda(to_tensor), Lambda(to_tensor)]
])

"""
from __future__ import division, print_function
import math
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import types
import collections
import torch

class JointCompose(object):
    """Composes several transforms together, support separate transformations for multiple input.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            if isinstance(t, collections.Sequence):
                assert isinstance(img, collections.Sequence) and len(img) == len(t), "size of image group and transform group does not fit"
                tmp_ = []
                for i, im_ in enumerate(img):
                    if callable(t[i]):
                        tmp_.append(t[i](im_))
                    else:
                        tmp_.append(im_)
                img = tmp_
            elif callable(t):
                img = t(img)
            elif t is None:
                continue
            else:
                raise Exception('unexpected type')                
        return img

class RandomScale(object):
    """Random resize the given PIL.Image(s)
    low: ratio of minimum size to original size
    high: ratio of maximum size to original size
    interpolation(s): interpolations used.
       IF auto, uses NEAREST neighbour for second input
    """

    def __init__(self, low, high, interpolations='auto'):
        self.low = low
        self.high = high
        self.interpolations = interpolations

    def __call__(self, images):
        single = False
        if not isinstance(images, collections.Sequence):
            images = [images]
            single = True
        interps = self.interpolations
        if interps == 'auto':
            interps = Image.BILINEAR
            if len(images) == 2:
                interps = [Image.BILINEAR, Image.NEAREST]
        if not isinstance(interps, collections.Sequence):
            interps = [interps] * len(images)
        resized = []
        ratio = random.uniform(self.low, self.high)
        for img, interp in zip(images, interps):
            h, w = img.size[0], img.size[1]
            h2, w2 = (int(ratio * h), int(ratio * w))
            img2 = img.resize((h2, w2), interp)
            resized.append(img2)
        if single:
            resized = resized[0]
        return resized

class Scale(object):
    # MONOCHANNEL FOR NOW # fixme
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, images):
        if random.random() < 0.5:
            single = False
            if not isinstance(images, collections.Sequence):
                images = [images]
                single = True
            images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
            if single:
                images = images[0]
        return images


def ensuretuple(inp, n=2):
    # duplicate value n times if needed
    if not isinstance(inp, collections.Sequence):
        inp = (inp,) * n
    assert len(inp) == n, "Expected input of size " + str(n)
    return inp

def pad_to_target(img, target_height, target_width, label=0):
    # Pad image with zeros to the specified height and width if needed
    # This op does nothing if the image already has size bigger than target_height and target_width.
    w, h = img.size
    left = top = right = bottom = 0
    doit = False
    if target_width > w:
        delta = target_width - w
        left = delta // 2
        right = delta - left
        doit = True
    if target_height > h:
        delta = target_height - h
        top = delta // 2
        bottom = delta - top
        doit = True
    if doit:
        img = ImageOps.expand(img, border=(left, top, right, bottom), fill=label)
    assert img.size[0] >= target_width
    assert img.size[1] >= target_height
    return img


class RandomCropPad(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    pad with pad_label if needed. auto -> (0/255)
    """

    def __init__(self, size, pad_label='auto'):
        self.target_height, self.target_width = ensuretuple(size)
        self.pad_label = pad_label

    def __call__(self, images):
        th, tw = self.target_height, self.target_width
        single = False
        if not isinstance(images, collections.Sequence):
            images = [images]
            single = True
        pad_label = self.pad_label
        if pad_label == 'auto':
            pad_label = 0
            if len(images) == 2:
                pad_label = [0, 255]
        returns = []
        for image, pad in zip(images, pad_label):
            image = pad_to_target(image, th, tw, pad)
            returns.append(image)
        w, h = returns[0].size
        for ret in returns[1:]:
            assert (w, h) == ret.size, "all inputs images must have same size"
        if w == tw and h == th:
            return returns

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return [ret.crop((x1, y1, x1 + tw, y1 + th)) for ret in returns]

class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    std is optional
    """

    def __init__(self, mean, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if self.std is None:
            for t, m in zip(tensor, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return tensor

class PILtoTensor(object):
    """ puts channels in front and convert to float, except if mode palette
    """

    def __init__(self):
        pass

    def __call__(self, inputs):
        single = False
        if not isinstance(inputs, collections.Sequence):
            inputs = [inputs]
            single = True
        res = []
        for im in inputs:
            if im.mode == 'P':
                dest = torch.from_numpy( np.array(im) )
                res.append( dest )
            else:
                dest = torch.from_numpy( np.array(im).transpose(2, 0, 1) )
                res.append( dest.float() )
        if single:
            res = res[0]
        return res

class TensortoPIL(object):
    """ Tensors to arrays
        With flat arrays: label with palette
        with 3d arrays: image, put first channel in the end
    """

    def __init__(self, color_map=None):
        self.color_map = color_map

    def __call__(self, inputs):
        single = False
        if not isinstance(inputs, collections.Sequence):
            inputs = [inputs]
            single = True
        res = []
        for tens in inputs:
            dest = tens.cpu().numpy()
            if dest.ndim == 3:
                dest = dest.transpose(1, 2, 0).astype(np.uint8)
                dest = Image.fromarray(dest)
            elif dest.ndim == 2:
                dest = dest.astype(np.uint8)
                dest = Image.fromarray(dest, "P")
                if self.color_map is not None:
                    cmap = [k for l in self.color_map for k in l]
                    dest.putpalette(cmap)
            res.append(dest)
        if single:
            res = res[0]
        return res
