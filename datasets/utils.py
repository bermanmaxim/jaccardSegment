#!/usr/bin/env python
# Martin Kersner, m.kersner@gmail.com
# 2016/03/11
#** Maxim Berman ** modified from https://github.com/martinkersner/train-DeepLab/

import scipy.io
import struct
import numpy as np
from PIL import Image

import sys

if 'ipykernel' in sys.modules:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def pascal_classes(with_background=True, with_void=True, reverse=False):
  classes = {'aeroplane' : 1,  'bicycle'   : 2,  'bird'        : 3,  'boat'         : 4,
             'bottle'    : 5,  'bus'       : 6,  'car'         : 7,  'cat'          : 8,
             'chair'     : 9,  'cow'       : 10, 'diningtable' : 11, 'dog'          : 12,
             'horse'     : 13, 'motorbike' : 14, 'person'      : 15, 'potted-plant' : 16,
             'sheep'     : 17, 'sofa'      : 18, 'train'       : 19, 'tv/monitor'   : 20}
  if with_background: classes['background'] = 0
  if with_void: classes['void'] = 255
  if reverse:
    return {v: k for k, v in classes.iteritems()}
  return classes

def pascal_palette(void=False):
  palette = {(  0,   0,   0) : 0 ,
             (128,   0,   0) : 1 ,
             (  0, 128,   0) : 2 ,
             (128, 128,   0) : 3 ,
             (  0,   0, 128) : 4 ,
             (128,   0, 128) : 5 ,
             (  0, 128, 128) : 6 ,
             (128, 128, 128) : 7 ,
             ( 64,   0,   0) : 8 ,
             (192,   0,   0) : 9 ,
             ( 64, 128,   0) : 10,
             (192, 128,   0) : 11,
             ( 64,   0, 128) : 12,
             (192,   0, 128) : 13,
             ( 64, 128, 128) : 14,
             (192, 128, 128) : 15,
             (  0,  64,   0) : 16,
             (128,  64,   0) : 17,
             (  0, 192,   0) : 18,
             (128, 192,   0) : 19,
             (  0,  64, 128) : 20 }
  if void:
    palette[(  224,  224, 192)] = 255

  return palette

def array_to_segmentation(array):
  array = array.astype(np.uint8)
  lab = Image.fromarray(array, "P")
  cmap = [k for l in color_map() for k in l]
  lab.putpalette(cmap)
  return lab

def pascal_palette_invert():
  palette_list = pascal_palette().keys()
  palette = ()
  
  for color in palette_list:
    palette += color

  return palette

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def pascal_mean_values():
  return np.array([103.939, 116.779, 123.68], dtype=np.float32)

def strstr(str1, str2):
  if str1.find(str2) != -1:
    return True
  else:
    return False

# Mat to png conversion for http://www.cs.berkeley.edu/~bharath2/codes/SBD/download.html
# 'GTcls' key is for class segmentation
# 'GTinst' key is for instance segmentation
def mat2png_hariharan(mat_file, key='GTcls'):
  mat = scipy.io.loadmat(mat_file, mat_dtype=True, squeeze_me=True, struct_as_record=False)
  return mat[key].Segmentation

def convert_segmentation_mat2numpy(mat_file):
  np_segm = load_mat(mat_file)
  return np.rot90(np.fliplr(np.argmax(np_segm, axis=2)))

def load_mat(mat_file, key='data'):
  mat = scipy.io.loadmat(mat_file, mat_dtype=True, squeeze_me=True, struct_as_record=False)
  return mat[key]

# Python version of script in code/densecrf/my_script/LoadBinFile.m
def load_binary_segmentation(bin_file, dtype='int16'):
  with open(bin_file, 'rb') as bf:
    rows = struct.unpack('i', bf.read(4))[0]
    cols = struct.unpack('i', bf.read(4))[0]
    channels = struct.unpack('i', bf.read(4))[0]

    num_values = rows * cols # expect only one channel in segmentation output
    out = np.zeros(num_values, dtype=np.uint8) # expect only values between 0 and 255

    for i in range(num_values):
      out[i] = np.uint8(struct.unpack('h', bf.read(2))[0])

    return np.rot90(np.fliplr(out.reshape((cols, rows))))

def convert_from_color_segmentation(arr_3d, use_void=False):
  arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
  palette = pascal_palette(use_void)
    
  for c, i in palette.items():
    m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
    arr_2d[m] = i

  return arr_2d

def create_lut(class_ids, max_id=256):
  # Index 0 is the first index used in caffe for denoting labels.
  # Therefore, index 0 is considered as default.
  lut = np.zeros(max_id, dtype=np.uint8)

  new_index = 1
  for i in class_ids:
    lut[i] = new_index
    new_index += 1

  return lut

def get_id_classes(classes):
  all_classes = pascal_classes()
  id_classes = [all_classes[c] for c in classes]
  return id_classes

def parallel_process(array, function, n_jobs=8, use_kwargs=False, front_num=3):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True,
            'smoothing': 0.1,
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out