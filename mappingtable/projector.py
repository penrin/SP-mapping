import os
import sys
import cv2
import numpy as np




class Projector():
    
    def __init__(self, filename):
        png = cv2.imread(filename)
        if png is None:
            raise Exception('%s could not read.' % filename)
        green = png[:, :, 1]
        i = np.where(green != 0)
        
        self.base_aspect = green.shape
        self.aspect = i[0].max() - i[0].min() + 1, i[1].max() - i[1].min() + 1
        self.upperleft = np.min(i[0]), np.min(i[1])
        self.filename = filename

    
    def get_areamap(self):
        i1 = self.upperleft[0]
        i2 = i1 + self.aspect[0]
        j1 = self.upperleft[1]
        j2 = j1 + self.aspect[1]
        p = np.zeros([self.base_aspect[0], self.base_aspect[1]], dtype=np.bool)
        p[i1:i2, j1:j2] = True
        return p

    
    def gen_canvas(self, dtype=np.uint8):
        return np.zeros([self.aspect[0], self.aspect[1], 3], dtype=dtype)


    def add_base(self, img):
        if (img.shape[0], img.shape[1]) != (self.aspect[0], self.aspect[1]):
            raise Exception('img aspect does not match.')
        i1 = self.upperleft[0]
        i2 = i1 + self.aspect[0]
        j1 = self.upperleft[1]
        j2 = j1 + self.aspect[1]
        if img.ndim == 3:
            base = np.zeros(
                    [self.base_aspect[0], self.base_aspect[1], img.shape[2]],
                    dtype=np.uint8)
            base[i1:i2, j1:j2, :] = img
        elif img.ndim == 2:
            base = np.zeros([self.base_aspect[0], self.base_aspect[1]], 
                            dtype=np.uint8)
            base[i1:i2, j1:j2] = img
        return base



def set_config(path2work):
    
    proj_list = []
    num = 0
    
    while True:
        num += 1
        filename = path2work + 'projector_%d.png' % num
        
        if os.path.isfile(filename):
            proj = Projector(filename)
            proj_list.append(proj)
        else:
            if num == 1:
                raise Exception('%s is not exists.' % filename)
            break
    
    return proj_list
    
    

def inspect_projectors(proj_list):
    
    if len(proj_list) == 1:
        return

    # base_aspect equality
    for i in range(1, len(proj_list)):
        if proj_list[0].base_aspect != proj_list[i].base_aspect:
            raise Exception('base_aspect is not equal')
             
    # overlap
    p_imgs = np.empty(
            [proj_list[0].base_aspect[0], proj_list[0].base_aspect[1], len(proj_list)],
            dtype=np.uint8
            )
    for i in range(len(proj_list)):
        p_imgs[:, :, i] = proj_list[i].get_areamap()
    p_sum = np.sum(p_imgs, axis=-1)
    overlap = np.sum(p_sum > 1)
    
    if overlap > 0:
        raise Exception('Projector regions overlap')

