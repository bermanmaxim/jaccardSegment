#!/usr/bin/env python
# Maxim Berman, bermanmaxim@gmail.com
# Load Pascal VOC + Berkeley extended annotations datasets

from __future__ import absolute_import, division, print_function
import os, sys
import scipy.io
from .common import Example, SegSet
if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
join = os.path.join
import warnings
from skimage.io import imread, imsave
from .utils import mat2png_hariharan, convert_from_color_segmentation, pascal_classes, color_map
import numpy as np
from shutil import copyfile
from PIL import Image
import platform

VOC_DIR = './VOCdevkit/VOC2012'
EXTENDED_DIR = './VOCdevkit/VOC2012/berkeley'

VOCVER = 'voc12'
CACHE_FUSE = join(VOC_DIR, 'FusedSets') # where to store the fused image lists
#
MASKS_DIR = join(VOC_DIR, 'SegmentationClassPalette')
pascal_classes = pascal_classes()
pascal_classes_inv = {v: k for k, v in pascal_classes.items()}

def load_extended_voc(voc_dir=VOC_DIR, extended_dir=EXTENDED_DIR,
        masks_dir=MASKS_DIR, cache_fuse=CACHE_FUSE, vocver=VOCVER):
    """
    Fuse VOC and Berkeley annotations
    Convert annotations to common 21-class + void palette png format
    Copy labels in same folder MASKS_DIR
    Returns train/val/test lists and classes (classes present on each image)
    Caches results in cache_fuse
    """

    trainaugf = join(cache_fuse, 'trainaug.txt')
    valaugf = join(cache_fuse, 'valaug.txt')
    testaugf = join(cache_fuse, 'testaug.txt')
    dirsf = join(cache_fuse, 'dirs.txt')
    infof = join(cache_fuse, 'info.txt')
    if (os.path.exists(cache_fuse)
           and os.path.isfile(trainaugf) and os.path.isfile(valaugf)
           and os.path.isfile(testaugf) and os.path.isfile(dirsf)
           and os.path.isfile(infof)):
        # load from cache if files exist
        train = [l.strip() for l in open(trainaugf)]
        val = [l.strip() for l in open(valaugf)]
        test = [l.strip() for l in open(testaugf)]
        [vocjpg, masks_dir] = [l.strip() for l in open(dirsf)]
        source = {}
        classes = {}
        with open(infof) as f:
            next(f) # skip header line
            for line in f:
                m = line.split()
                source[m[0]] = m[1]
                classes[m[0]] = m[2:]
        print("Loaded dataset from cache " + cache_fuse)
    else: # no cache, do the computations
        # fuse image lists
        vocsets = join(voc_dir, 'ImageSets', 'Segmentation')
        augmented_root = join(extended_dir, 'benchmark_RELEASE', 'dataset')

        voctrainF = join(vocsets, 'train.txt')
        vocvalF = join(vocsets, 'val.txt')
        voctestF = join(vocsets, 'test.txt')

        augtrainF = join(augmented_root, 'train.txt')
        augvalF = join(augmented_root, 'val.txt')

        voctrain = [l.strip() for l in open(voctrainF)]
        val = [l.strip() for l in open(vocvalF)]
        test = [l.strip() for l in open(voctestF)]
        augtrain = [l.strip() for l in open(augtrainF)]
        augval = [l.strip() for l in open(augvalF)]

        source = {}
        for im in augtrain + augval:
            source[im] = 'aug'
        for im in voctrain + val + test:
            source[im] = str(vocver)

        train = sorted(set(augtrain + voctrain + augval) - set(val) - set(test))
        print("Loaded image sets, {} train / {} val / {} test"
            .format(len(train), len(val), len(test)))

        # convert to common format
        vocjpg = join(voc_dir, 'JPEGImages')
        vocseg = join(voc_dir, 'SegmentationClass')
        augseg = join(augmented_root, 'cls')

        if not os.path.exists(masks_dir):
            os.makedirs(masks_dir)

        print('Converting masks to common format...')
        classes = {}
        # just copy voc labels and scan classes
        for im in tqdm([im for im in train + val if source[im] != 'aug'],
                        desc='VOC: copy labels...'):
            srcf = join(vocseg, im+'.png')
            copyfile(srcf, join(masks_dir,  im + '.png'))
            array = np.array(Image.open(srcf))
            clsuniques = np.unique(array)
            classes[im] = [pascal_classes_inv[k] for k in clsuniques]
            # src = imread(srcf)
            # img = convert_from_color_segmentation(src, use_void=True)
            # clsuniques = np.unique(img)
            # classes[im] = [pascal_classes_inv[k] for k in clsuniques]
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            #     imsave(join(masks_dir,  im + '.png'), img)

        # MAT labels to png
        cmap = color_map(255)
        flat_cmap = [i for l in cmap for i in l]
        for im in tqdm([im for im in train + val if source[im] == 'aug'],
                        desc='AUG: MAT to 1D PNG...'):
            srcf = join(augseg, im+'.mat')
            img = mat2png_hariharan(srcf)
            clsuniques = np.unique(img)
            classes[im] = [pascal_classes_inv[k] for k in clsuniques]
            newimg = Image.fromarray(img, mode="P")
            newimg.putpalette(flat_cmap)
            newimg.save(join(masks_dir,  im + '.png'))
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            #     imsave(join(masks_dir,  im + '.png'), img)

        # memoize everything
        if not os.path.exists(cache_fuse):
            os.makedirs(cache_fuse)
        open(join(cache_fuse, trainaugf), 'w').write("\n".join(train))
        open(join(cache_fuse, valaugf), 'w').write("\n".join(val))
        open(join(cache_fuse, testaugf), 'w').write("\n".join(test))
        dirs = [vocjpg, masks_dir]
        open(join(cache_fuse, dirsf), 'w').write("\n".join(dirs))
        with open(infof, 'w') as f:
            f.write('\t'.join(['name', 'source', 'classes...']) + '\n')
            for im in train + val:
                f.write(im + '\t' + source[im] + '\t' + '\t'.join(classes[im]) + '\n')
        print("Saved cache in " + cache_fuse)

    train = SegSet('AugVocTrain',
                   [Example(im, source[im], classes[im]) for im in train],
                   vocjpg,
                   pascal_classes,
                   masks_dir,
                  )
    val = SegSet('AugVocVal',
                 [Example(im, source[im], classes[im]) for im in val],
                 vocjpg,
                 pascal_classes,
                 masks_dir,
                )
    test = SegSet('AugVocTest',
                  [Example(im, vocver) for im in test],
                  vocjpg,
                  pascal_classes,
                 )
    return train, val, test


if __name__ == '__main__':
    train, val, test = load_extended_voc()


