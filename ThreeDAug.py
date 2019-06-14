import time
import numpy as np
import os
import random
from itertools import product

from multiprocessing.pool import ThreadPool, Pool
from multiprocessing import current_process, cpu_count

import cv2
from skimage.transform import rotate
from PIL import Image

from tools import load_imgs, get_imgs, getFalsePoints, cubeEm

class ThreeDAug:
    def __init__(self, image, mask = np.array([None]), pool_size = 1):
        self.params = {}
        self.random_rotate = 0
        self.image = image
        self.mask = mask
        self.pool_size = pool_size

        self.w = self.image.shape[1]
        self.h = self.image.shape[0]
          
    def get_img(self):
        return self.image
        
    def add_rotate(self, angle=90, scale=1.0):
        self.params['rotate'] = [angle, scale]
        
    def add_flip(self, vflip=True, hflip=True):
        self.params['flip'] = [vflip, hflip]

    def start_process(self):
        print ('Starting process ...', current_process().name)

    def process(self, verbose = False):
        if len(self.params) < 1:
            print("You first need to add parameters (i.e. add_flip or add_rotate)")
        else:
            for p in self.params.items():
                if p[0] == 'rotate':
                    self.random_rotate = random.randint(1, self.params['rotate'][0])
                    if verbose:
                        print("\nPerforming rotation of", self.random_rotate, "degrees ...")
                        tick = time.clock()
                    pool = Pool(processes=self.pool_size, initializer=self.start_process)
                    image = pool.map(self.rotateIt, self.image)
                    self.image = np.asarray(image)
                    pool.close() # no more tasks
                    pool.join()  # wrap up current tasks
                    if verbose:
                        tock = time.clock()
                        print("   ... completed in", tock-tick, "seconds.\n")                    
                        
                if p[0] == 'flip':
                    pass
                    #print("Performing flip ...")

    def rotateIt(self, image):
        '''
        Rotate the image
        :param image: image to be processed
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''

        #rotate matrix
        M = cv2.getRotationMatrix2D((self.w/2,self.h/2), self.random_rotate, self.params['rotate'][1])
        #rotate
        image = cv2.warpAffine(image,M,(self.w,self.h))
        
        #im = Image.fromarray(image)
        #im.rotate(self.params['rotate'][0], Image.BICUBIC, expand=True)
        
        return np.array(image)

    def flipIt(self, image, vflip=False, hflip=False):
        '''
        Flip the image
        :param image: image to be processed
        :param vflip: whether to flip the image vertically
        :param hflip: whether to flip the image horizontally
        '''
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image 
            

#image = load_png(r'F:\Data\MILDBL\mild_extracted_images\MILD_ - 128CM290341')
            
#ThreeDAug('F:\Data\MILDBL\mild_extracted_images\MILD_ - 101AR300932')

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    path = r'F:\Data\MILDBL\mild_extracted_images\MILD_ - 128CM290341'
    
    img, mask = get_imgs(path, has_mask = True)
    #points = getFalsePoints(mask, 4)
    #cubes = cubeEm(img, points)
    #tda = ThreeDAug(cubes[0])     
    #cube = tda.get_img()
    #tda.add_rotate(angle = 15)
    #tda.add_flip()
    #tda.process(verbose = True)
    #new_cube = tda.get_img()
    #plt.imshow(new_cube[25, :, :])
    
    tda = ThreeDAug(img)
    tda.add_rotate(angle = 15)    
    tda.process(verbose = True)            
            
    
    
    
    

    
    
    
    
    
    
            
            
            