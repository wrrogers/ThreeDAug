# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:58:54 2019

@author: wrrog
"""

import os
import numpy as np
from tools import get_imgs
from PointsImageMask import PointsImageMask as PIM

class Generator:
    def __init__(self, path, batch_size, points_list=None, z =  50, rad = 50):
        self.path = path
        self.points_list = points_list
        self.batch_size = batch_size
        self.z = z
        self.rad = rad
        self.params = {}
        self.points = [[225, 367, 310], [387,177,250], [319, 387, 227], [88, 310, 171]]
        
    def generate(self):
        while True:
            count = 0
            scan_list = [folder for folder in os.listdir(self.path) if folder[-4:] != '.*']
            all_cubes = np.empty((self.batch_size, self.z, self.rad*2, self.rad*2))
            for n in range(self.batch_size):
                id = np.random.choice(scan_list, 1)
                
                print("The ID:", id)
                ipath = os.path.join(self.path, id)
                print(ipath)
                
                img, mask = get_imgs(ipath)
                cubes = PIM(self.points, img, mask, self.z, self.rad)
                
                
                func = getattr(cubes, 'bar')
                func()
                
                for c in range(cubes.shape[0]):
                    count+=1
                    all_cubes[count] = cubes[c]
                    if count == self.batch_size:
                        break
                
                if count == self.batch_size:
                    break
                
            yield all_cubes
            
    def generateFirst(self):
        scan_list = [folder for folder in os.listdir(self.path) if folder[-4:] != '.*']
        id = np.random.choice(scan_list, 1)[0]
        
        print("\nThe ID:", id)
        ipath = os.path.join(self.path, id)
        print("\n", ipath)
        
        img, mask = get_imgs(ipath)
        cubes = PIM(self.points, img, mask, self.z, self.rad)
        
        print("Image width:", cubes.w, "height:", cubes.h)
        
        for p in self.params.items():
            print("\nDoing ...", p[0], "\n")
            func = getattr(cubes, "add_"+p[0])
            func(*p[1])
        
        cubes.process()
        
        return cubes.get_cubes()
        
    def add_rotate(self, angle=90, scale=1.0):
        self.params['rotate'] = [angle, scale]
        
    def add_flip(self, vflip=True, hflip=True):
        self.params['flip'] = [vflip, hflip]
        
    def add_shift(self, shift = True):
        self.params['shift'] = [shift]

    def add_elastic(self, alpha = 3, sigma = 0.07, alpha_affine = 0.09):
        self.params['elastic'] = [alpha*self.rad*2, sigma*self.rad*2, alpha_affine*self.rad*2, 10000]

    def add_noise(self, modes = None, proba = .5):
        noises = ["gaussian", "localvar", "poisson", "salt", "pepper", "s&p", "speckle"]
        if modes == 'all' or modes == ['all'] or modes == None:
            self.params['noise'] = [noises, proba]
        else:
            for mode in modes:
                if mode not in noises:
                    print("Error:", mode, " is not a supported.", \
                          "the supported modes:", noises)
                    break
                else:
                    self.params['noise'] = [modes, proba]
    
    def add_zoom(self, pxl = 6, proba = .5):
        self.params['zoom'] = [pxl, proba]
            
    def get_params(self):
        print(" -Parameter-    -Variables\n", \
              "------------------------------")
        for param in self.params.items():
            print("  {:<10}\t{}".format(param[0], param[1]))

        
        
if __name__ == "__main__":

    path = r'F:\Data\MILDBL\mild_extracted_images'
    tdg = Generator(path, 32)
    tdg.add_zoom()
    tdg.get_params()
    cubes = tdg.generateFirst()

    print(cubes.shape)



