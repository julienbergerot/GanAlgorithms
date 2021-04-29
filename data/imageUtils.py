import numpy as np
import scipy
import random
from PIL import Image
import os

class Rotate :
    def __init__(self, angle_range=(0,360), axes=(0, 1), mode='reflect', random_state=np.random):
        assert isinstance(angle_range, tuple)
        self.angle_range = angle_range
        self.random_state = random_state
        self.axes = axes
        self.mode = mode
        
    def __call__(self,*images) :
        angle = random.uniform(self.angle_range[0],self.angle_range[1])

        result = []
        for image in images :            
            image = scipy.ndimage.interpolation.rotate(image, angle, reshape=False, axes=self.axes, mode=self.mode)
            result.append(image)
        if len(result) == 1 :
            return result[0]
        return result

class Crop :
    def __init__(self, size):
        if isinstance(size,int):
            self.size = (int(size), int(size))
        else:
            self.size = (int(size[0]), int(size[1]))
    
    def __call__(self,*images) :
        h, w = images[0].shape[:2]
        th, tw = self.size
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        result = []
        for image in images :
            if w == tw and h == th:
                result.append(image)
            else :
                result.append(image[y1: y1 + th, x1:x1 + tw, :])
        if len(result) == 1 :
            return result[0]
        return result

class CenterCrop :
    def __init__(self,size) :
        if isinstance(size,int):
            self.size = (int(size), int(size))
        else:
            self.size = (int(size[0]), int(size[1]))
    

    def __call__(self,*images) :
        h, w = images[0].shape[:2]
        th, tw = self.size
        x1 = w // 2 - tw // 2
        y1 = h //2 - th // 2

        result = []
        for image in images :
            if w == tw and h == th:
                result.append(image)
            else :
                result.append(image[y1: y1 + th, x1:x1 + tw, :])
        if len(result) == 1 :
            return result[0]
        return result

class Norm :
    def __init__(self,mode=None,*args) :
        self.mode = mode
        self.args = args # two values, either min-max or mean-std

    def __call__(self,array) :
        if self.mode == "8bits" :
            return (array-127.5) / 127.5
        if self.mode == "12bits" :
            return array / 4096 * 2 - 1
        if self.mode == "16bits" :
            return array / 65536 * 2 - 1
        if self.mode == "meanStd" :
            return (array-self.args[0]) / self.args[std]
        if self.mode == "minMax" :
            return (array - self.args[0]) / (self.args[1] - self.args[0]) * 2 - 1

class SaveImg :
    def __init__(self,path,mode=None) :
        self.path = os.path.join(path,"outputs")
        self.count = 0
        self.mode = mode

        if not os.path.isdir(self.path) :
            os.mkdir(self.path)
    
    def __call__(self,*images,mode=None) :
        if mode != None :
            self.mode = mode
        for image in images :
            if self.mode == "[-1,1]" :
                image = (image + 1) / 2
            image = Image.fromarray(image)
            image.save(os.path.join(self.path,"{}.png".format(str(self.count))))
            self.count += 1