import os
import sys
import cv2
import numpy as np






class Projector():
    
    def __init__(self, filename):
        png = cv2.imread(filename)
        if png is None:
            print(filename, 'could not read.')
            sys.exit()
        green = png[:, :, 1]
        i = np.where(green != 0)
        
        self.base_aspect = green.shape
        self.aspect = i[0].max() - i[0].min() + 1, i[1].max() - i[1].min() + 1
        self.upperleft = np.min(i[0]), np.min(i[1])
        self.filename = filename

    
    def get_image(self):
        i1 = self.upperleft[0]
        i2 = i1 + self.aspect[0]
        j1 = self.upperleft[1]
        j2 = j1 + self.aspect[1]
        p = np.zeros([self.base_aspect[0], self.base_aspect[1]], dtype=np.bool)
        p[i1:i2, j1:j2] = True
        return p
        


class IntegratedProjector():
    
    def __init__(self, proj_list):
        self.proj_list = proj_list
        self._inspect()

    def _inspect(self):
        p = self.proj_list
        if len(self.proj_list) == 1:
            return

        # base_aspect equality
        for i in range(1, len(p)):
            if p[0].base_aspect != p[i].base_aspect:
                print('base_aspect is not equal.')
                print(p[i].filename)
                sys.exit()
        
        # overlap
        p_imgs = np.empty(
                [p[0].base_aspect[0], p[0].base_aspect[1], len(p)],
                dtype=np.uint8
                )
        for i in range(len(p)):
            p_imgs[:, :, i] = p[i].get_image()
        p_sum = np.sum(p_imgs, axis=-1)
        overlap = np.sum(p_sum > 1)
        
        if overlap > 0:
            print('Projector regions overlap.')
            sys.exit()





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
                print(filename, 'is not exits.')
                print('Please put the projector cofiguration png.')
                sys.exit()
            break
    
    # integration
    proj_integ = IntegratedProjector(proj_list)

    return proj_integ
    
    
    



