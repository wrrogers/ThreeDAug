import time
import numpy as np
import os
import sys
import random
from itertools import product

from multiprocessing.pool import ThreadPool, Pool
from multiprocessing import current_process, cpu_count

import cv2
from skimage.transform import rotate
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances
from tools import load_imgs, get_imgs

import traceback

class ThreeDAug:
    def __init__(self, points, image, mask, z, rad, pool_size = 1):
        self.params = {}
        self.random = {'rotate' : 0, 'vflip' : 'True', 'hflip' : 'True'}
        self.points = points
        self.f_points = self.getFalsePoints(mask, len(points), points)
        self.image = image
        self.mask = mask
       
        self.z = z
        self.rad = rad
        
        self.pool_size = pool_size

        self.w = self.image.shape[2]
        self.h = self.image.shape[1]
        
        self.processed_cubes = True
        
    def get_cubes(self):
        if self.processed_cubes:
            print("Cubes have not been processed yet")
        else:
            return self.processed_cubes, self.processed_false_cubes
            
    def add_rotate(self, angle=90, scale=1.0):
        self.params['rotate'] = [angle, scale]
        
    def add_flip(self, vflip=True, hflip=True):
        self.params['flip'] = [vflip, hflip]

    def add_elastic(self, alpha = 3, sigma = 0.07, alpha_affine = 0.09):
        self.params['elastic'] = [alpha*self.rad*2, sigma*self.rad*2, alpha_affine*self.rad*2, 10000]

    def start_process(self):
        print ('Starting process ...', current_process().name)

    def process(self, verbose = False):
        cubes = self.cubeEm(self.points)
        f_cubes = self.cubeEm(self.f_points)
        self.processed_cubes, self.processed_false_cubes = [], []
        if verbose:
            print("\nProcessing cubes with parameters:", list(self.params.keys()), "...")
        for c, fc in zip(cubes, f_cubes):
            tick = time.clock()
            processed_cube = self.process_cube(c, verbose = verbose)
            tock = time.clock()
            if verbose:
                print("   ... true point done in", tock - tick, "seconds.")
            self.processed_cubes.append(processed_cube)
            tick = time.clock()
            processed_cube = self.process_cube(fc, verbose = verbose)
            tock = time.clock()
            if verbose:
                print("   ... false point done in:", tock - tick, "seconds.")            
            self.processed_false_cubes.append(processed_cube)

    def process_cube(self, image, verbose = False):
        '''
        Process all image transformations
        '''

        if len(self.params) < 1:
            print("You first need to add parameters (i.e. add_flip or add_rotate)")
        else:
            for p in list(self.params.keys()):
                
                if p == 'rotate':
                    self.random['rotate'] = random.randint(1, self.params['rotate'][0])
                    if verbose:
                        print("\n      Performing rotation of", self.random['rotate'], "degrees ...")
                        tick = time.clock()
                    
                    rotated_image = np.empty(image.shape)
                    for n, slice in enumerate(image):
                            rotated_image[n,:,:] = self.rotateIt(slice)
                    image = rotated_image
                    
                    if verbose:
                        tock = time.clock()
                        print("         ... completed in", round(tock-tick, 4), "seconds.")                    
                        
                if p == 'flip':
                    if self.params['flip'][0]:
                        self.random['vflip'] = bool(random.randint(0, 1))
                    else:
                        self.random['vflip'] = False
                    if self.params['flip'][1]:
                        self.random['hflip'] = bool(random.randint(0, 1))
                    else:
                        self.random['hflip'] = False
                    if verbose:
                        print("\n      Performing flips horizontally: {} and vertically: {}.".format( self.random['hflip'], self.random['vflip'] ))
                        tick = time.clock()
                    if self.random['vflip'] != False or self.random['hflip'] != False:
                        
                        flipped_image = np.empty(image.shape)
                        for n, slice in enumerate(image):
                            flipped_image[n, :, :] = self.flipIt(slice)
                        image = flipped_image
                        
                    if verbose:
                        tock = time.clock()
                        print("         ... completed in", round(tock-tick, 4), "seconds.")              

                if p == 'elastic':
                    random_int = random.randint(1, self.params['elastic'][3])
                    if verbose:
                        print("\n      Performing elastic transormation ({})...".format(random_int))
                        tick = time.clock()
                    image = self.elastIt(image, random_int)
                    if verbose:
                        tock = time.clock()
                        print("         ... completed in", round(tock-tick, 4), "seconds.\n")                       
                
            return image

    def rotateIt(self, image):
        '''
        Rotate the image
        :param image: image to be processed
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''

        #rotate matrix
        M = cv2.getRotationMatrix2D((self.rad*2,self.rad*2), self.random['rotate'], self.params['rotate'][1])
        #rotate
        image = cv2.warpAffine(image,M,(self.rad*2,self.rad*2))
        
        image = np.array(image)
        #im = Image.fromarray(image)
        #im.rotate(self.params['rotate'][0], Image.BICUBIC, expand=True)

        return image

    def flipIt(self, image):
        '''
        Flip the image
        :param image: image to be processed
        :param vflip: whether to flip the image vertically
        :param hflip: whether to flip the image horizontally
        '''
        if self.random['hflip'] or self.random['vflip']:
            if self.random['hflip'] and self.random['vflip']:
                c = -1
            else:
                c = 0 if self.random['vflip'] else 1
            image = cv2.flip(image, flipCode=c)
        return image

    def elastIt(self, image, random_int):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
             Convolutional Neural Networks applied to Visual Document Analysis", in
             Proc. of the International Conference on Document Analysis and
             Recognition, 2003.
    
         Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
        """
        random_state = np.random.RandomState(random_int)
        
        shape = image.shape
        shape_size = shape[:2]
        
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + random_state.uniform(-self.params['elastic'][2], self.params['elastic'][2], size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.params['elastic'][1]) * self.params['elastic'][0]
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.params['elastic'][1]) * self.params['elastic'][0]
        #dz = np.zeros_like(dx)
    
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    
        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    
    def getFalsePoints(self, mask, num_cubes, true_pos = []):
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
    
    def cubeEm(self, points):
        points = np.array(points)
        cubes = np.empty((points.shape[0], self.z, self.rad*2, self.rad*2))
        for n, point in enumerate(points):
            
            # Fix points that are above or below the border in any direction
            for m, p in enumerate(point):
                if p + self.rad > self.image.shape[m]:
                    points[n, m] = self.image.shape[m] - self.rad
                if p - self.rad < 0:
                    points[n, m] = self.rad
                    
            cube = self.image[point[0] - int(self.z/2): point[0] + int(self.z/2),
                              point[1] - self.rad: point[1] + self.rad,
                              point[2] - self.rad: point[2] + self.rad]

            cubes[n] = cube
        return cubes

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    path = r'F:\Data\MILDBL\mild_extracted_images\MILD_ - 109BG060445'
    
    points = [[225, 367, 310], [387,177,250], [319, 387, 227], [88, 310, 171]]
    
    img, mask = get_imgs(path, has_mask = True)
    
    tda = ThreeDAug(points, img, mask, z = 50, rad = 25)
    tda.add_rotate(angle = 15)
    tda.add_flip(vflip=True, hflip=True)    
    tda.add_elastic()
    tda.process(verbose = True)
    #new_img = tda.get_img()
    #plt.imshow(new_img[150,:,:])
    