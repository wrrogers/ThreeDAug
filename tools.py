# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:53:23 2019

@author: wrrog
"""

import os
import numpy as np
import random
import time

from multiprocessing.pool import ThreadPool, Pool
from multiprocessing import current_process, cpu_count

from sklearn.metrics.pairwise import euclidean_distances

from PIL import Image

def start_process():
    print ('Starting process ...', current_process().name)

def load_imgs(files, type = 'image', pool_size = 1):
    pool = Pool(processes=pool_size, initializer=start_process)
    img = pool.map(Image.open,files)
    pool.close() # no more tasks
    pool.join()  # wrap up current tasks
    img = [np.array(i) for i in img]
    img = np.asarray(img)
    if type == 'mask':
        img = img / 255.0
        img = img.astype('int8')
    return img

def get_imgs(path, extension = 'png', has_mask = True, 
             pool_size = 1, verbose = True,
             image_prefix = '', image_suffix = '_i', 
             mask_prefix  = '', mask_suffix  = '_m'):
    '''
    Import image
    :param path: Path to the image
    :param image_name: image name
    '''
    
    has_mask = has_mask
    path = path
    files = os.listdir(path)
    image_prefix = image_prefix
    image_suffix = image_suffix
    extension = extension
    end   = (len(image_suffix) + 1 + len(extension)) *-1
    start = (len(image_prefix)) * -1
    image_files = [f for f in files if f[end:].upper() == "{}.{}".format(image_suffix, extension).upper()]
    image_files = [f for f in image_files if f[:start].upper() == image_prefix.upper()]
    image_files = [os.path.join(path, f) for f in image_files]
    
    if has_mask:
        mask_prefix = mask_prefix
        mask_suffix = mask_suffix
        end   = (len(mask_suffix) + 1 + len(extension)) *-1
        start = (len(mask_prefix)) * -1
        mask_files = [f for f in files if f[end:].upper() == "{}.{}".format(mask_suffix, extension).upper()]
        mask_files = [f for f in mask_files if f[:start].upper() == mask_prefix.upper()]
        mask_files = [os.path.join(path, f) for f in mask_files]
    
    if verbose:
        print("\nLoading image ...")
        tick = time.clock()
    image = load_imgs(image_files, pool_size = pool_size)
    if verbose:
        tock = time.clock()
        print("   ... image loaded in", tock - tick, "seconds.")
    if has_mask:
        if verbose:
            print("\nLoading mask ...")
            tick = time.clock()        
        mask = load_imgs(mask_files, pool_size = pool_size, type = 'mask')
        if verbose:
            tock = time.clock()
            print("   ... mask loaded in", tock - tick, "seconds.\n")     
        return image, mask
    
    return image

def getFalsePoints(mask, num_cubes, true_pos = [], rad = 25):
    #middle = 0
    points = []
    #sums = []
    for n in range(num_cubes):
        while True:
            rand_x = random.randint(0, mask.shape[2]-1)
            rand_y = random.randint(0, mask.shape[1]-1)
            rand_z = random.randint(0, mask.shape[0]-1)

            if len(true_pos) > 0:
                distances = euclidean_distances(true_pos, [[rand_x, rand_y, rand_z]])
                if np.min(distances) < 100:
                    continue
                
            #proba = random.randint(0,100)/100

            #sums.append(np.sum(mask[rand_z-rad:rand_z+rad, rand_y-rad:rand_y+rad, rand_x-rad:rand_x+rad]))
            #print("The sum is:", sums[n])
            #print("The random point is:", mask[rand_z, rand_y, rand_x])

            if mask[rand_z, rand_y, rand_x] == 1:
                #if np.sum(mask[rand_z-rad:rand_z+rad, rand_y-rad:rand_y+rad, rand_x-rad:rand_x+rad]) == 125000:    
                #    middle += 1
                point = [rand_z, rand_y, rand_x]
                points.append(point)
                break
            
    return points#, middle, np.array(sums)

def cubeEm(img, points, rad = 25):
    cubes = []
    for point in points:
        cubes.append( img[point[0] - rad: point[0] + rad,
                          point[1] - rad: point[1] + rad,
                          point[2] - rad: point[2] + rad] )
    cubes = np.asarray(cubes)        
    return cubes




















