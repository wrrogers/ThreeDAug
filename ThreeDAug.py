import numpy as np # linear algebra
import os
import cv2
from PIL import Image
import random
from multiprocessing.pool import ThreadPool, Pool

class ThreeDAug:
    def __init__(self, path, extension = 'png', has_mask = True,
                 image_prefix = '', image_suffix = '_i', 
                 mask_prefix  = '', mask_suffix  = '_m'):
        '''
        Import image
        :param path: Path to the image
        :param image_name: image name
        '''
        self.params = {}
        path = path
        files = os.listdir(path)
        image_prefix = image_prefix
        image_suffix = image_suffix
        extension = extension
        end   = (len(image_suffix) + 1 + len(extension)) *-1
        start = (len(image_prefix)) * -1
        image_files = [f for f in files if f[end:].upper() == "{}.{}".format(image_suffix, extension).upper()]
        image_files = [f for f in image_files if f[:start].upper() == image_prefix.upper()]
        self.image = self.load_imgs(image_files)
        
        if has_mask:
            mask_prefix = mask_prefix
            mask_suffix = mask_suffix
            end   = (len(mask_suffix) + 1 + len(extension)) *-1
            start = (len(mask_prefix)) * -1
            mask_files = [f for f in files if f[end:].upper() == "{}.{}".format(mask_suffix, extension).upper()]
            mask_files = [f for f in mask_files if f[:start].upper() == mask_prefix.upper()]
            self.mask = self.load_mask(mask_files, type = 'mask')
        
        #print(image_files)
        #print(mask_files)
        #self.image = cv2.imread(path+image_name)
        
    def load_imgs(self, files, type = 'image'):
        pool = Pool()
        img = pool.map(Image.open,files)
        img = [np.array(i) for i in img]
        img = np.asarray(img)
        if type == 'mask':
            img = int(img / 255)
        return img
    
    def get_img(self):
        return self.image
    
    def get_mask(self):
        return self.mask
        
    def rotate(self, angle=90, scale=1.0):
        self.params['rotate'] = [angle, scale]

    def rotateIt(self):
        '''
        Rotate the image
        :param image: image to be processed
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''
        w = image.shape[1]
        h = image.shape[0]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
        #rotate
        image = cv2.warpAffine(image,M,(w,h))
        return image

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
            
            
            
ThreeDAug(r'F:\Data\MILDBL\mild_extracted_images\MILD_ - 101AR300932')
            
            
            
            
            
            
            
            
            
            
            