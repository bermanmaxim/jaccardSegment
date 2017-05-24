from __future__ import absolute_import, division, print_function
from scipy.misc import imread
from PIL import Image
import numpy as np
import os

class SegSet(object):
    # Collection of segmentation Examples
    def __init__(self, name, examples, imagespath, classes, maskspath=None):
        self.name = name
        self.imagespath = imagespath
        self.maskspath = maskspath
        self.examples = examples
        self.classes = classes
        super(SegSet, self).__init__()
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, key):
        if isinstance( key, slice ) :
            if key.start == key.stop == key.step == None:
                token = ""
            else:
                token = "["
                if key.start != None: token += str(key.start)
                token += ":"
                if key.stop != None: token += str(key.stop)
                if key.step != None: token += ":" + str(key.step)
                token += "]"
            return SegSet(self.name + token, self.examples[key], self.imagespath, self.classes, self.maskspath)
        if isinstance( key, list ) :
            # list of indices
            token = "[<list>]"
            return SegSet(self.name + token,
                          [self.examples[i] for i in key],
                          self.imagespath, self.classes, self.maskspath)
        elif isinstance( key, int ) :
            return self.examples[key]
        elif isinstance( key,  str ): # select a category
            token = "[" + key + "]"
            if key[0] == '~':  # select complementary
                key = key[1:]
                selected = [ex for ex in self.examples if key not in ex.classes]
            else:
                selected = [ex for ex in self.examples if key in ex.classes]
            return SegSet(self.name + token, selected, self.imagespath, self.classes, self.maskspath)
        else:
            raise TypeError, "Invalid argument type."
    def __repr__(self):
        return ("<{}: collection of {} examples>".format(
                      self.name, 
                      len(self.examples))
               )
    def __add__(self, other):
        assert self.imagespath == other.imagespath
        if self.maskspath and other.maskspath:
            assert self.maskspath == other.maskspath
        maskspath = self.maskspath if self.maskspath else other.maskspath
        return SegSet(self.name + "+" + other.name,
                      self.examples + other.examples,
                      self.imagespath,
                      self.classes,
                      maskspath)
    def impath(self, example):
        example = self.examples[example] if isinstance(example, int) else example
        return os.path.join(self.imagespath, example.name + ".jpg")
    def maskpath(self, example):
        example = self.examples[example] if isinstance(example, int) else example
        return os.path.join(self.maskspath, example.name + ".png")
    def imread(self, example, kind="scipy"):
        ipath = self.impath(example)
        if kind == "scipy":
            return imread(ipath)
        im = Image.open(ipath)
        if kind == "PIL":
            return im
        if kind == "array":
            return np.array(im)
    def binarize(self, cls):
        token = ".binarize({})".format(cls)
        binarizedset = BinarizedSegSet(self.name + token, self.examples,
                                       self.imagespath, self.classes, cls, self.maskspath)
        return binarizedset
    def maskread(self, example, kind="array"):
        mpath = self.maskpath(example)
        im = Image.open(mpath)
        if kind == "PIL":
            return im
        elif kind == "array":
            return np.array(im)
        else:
            raise NotImplementedError("Unknown return kind {}".format(kind))
#        return imread(mpath)
    def read(self, example, kind="array"):
        return self.imread(example, kind), self.maskread(example, kind)
        

class BinarizedSegSet(SegSet):
    def __init__(self, name, examples, imagespath, classes, target, maskspath=None):
        self.target = target
        super(BinarizedSegSet, self).__init__(name, examples, imagespath, classes, maskspath)
    def maskread(self, example, kind="array", withvoid=True):
        example = self.examples[example] if isinstance(example, int) else example
        mpath = self.maskpath(example)
        im = Image.open(mpath)
        if self.target in example.classes:
            target_idx = self.classes[self.target]
            arr = np.array(im)
            mask = arr == target_idx
            voidmask = arr == self.classes['void']
            arr[mask] = 1
            arr[~mask] = 0
            arr[voidmask] = self.classes['void']
        else:
            # return 0 labels
            arr = np.array(im)
            arr.fill(0)
        if kind == "array":
            return arr
        elif kind == "PIL":
            im = Image.fromarray(arr, "P")
            im.putpalette([0, 0, 0, 255, 255, 255]
                           + [255, 255, 178] * 253
                           + [255, 178, 253])
            return im
        else:
            raise NotImplementedError("Unknown return kind {}".format(kind))
    def binarize(self, cls):
        raise NotImplementedError("Already binarized to", self.target)
    def __add__(self, other):
        assert self.imagespath == other.imagespath
        if self.maskspath and other.maskspath:
            assert self.maskspath == other.maskspath
        maskspath = self.maskspath if self.maskspath else other.maskspath
        assert self.target == other.target
        return BinarizedSegSet(self.name + "+" + other.name,
                      self.examples + other.examples,
                      self.imagespath,
                      self.classes,
                      self.target,
                      maskspath)



class Example(object):
    def __init__(self, name, source, classes=[]):
        self.name = name
        self.source = source
        self.classes = classes
    def __repr__(self):
        return ("<Example {}>".format(self.name))

