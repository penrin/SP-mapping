import sys
import os
import cv2
import numpy as np



class Screen():
    
    def __init__(self, filename):
        
        png = cv2.imread(filename)
        if png is None:
            raise Exception('%s could not read.' % filename)
        self.png = png
        
        self.area_polar, self.area_azimuth = self._projection_area()

    
    def _projection_area(self):
        blue = self.png[:, :, 0]
        green = self.png[:, :, 1]
        screen = np.logical_or(blue, green) 
        ij_screen = np.where(screen)
        polar = [
                np.min(ij_screen[0]) / screen.shape[0] * 180,
                (np.max(ij_screen[0]) + 1) / screen.shape[0] * 180
                ]
        azimuth = [
                np.min(ij_screen[1]) / screen.shape[1] * 360,
                (np.max(ij_screen[1]) + 1) / screen.shape[1] * 360
                ]
        return polar, azimuth
    
    
    def get_projection_area(self, image_HW):
        # image_HW: pixel size of the caputured graypattern image
        i1 = int(np.floor(image_HW[0] * self.area_polar[0] / 180))
        i2 = int(np.ceil(image_HW[0] * self.area_polar[1] / 180))
        j1 = int(np.floor(image_HW[1] * self.area_azimuth[0] / 360))
        j2 = int(np.ceil(image_HW[1] * self.area_azimuth[1] / 360))
        return i1, i2, j1, j2


    def get_masked_index(self, image_HW):
        i1, i2, j1, j2 = self.get_projection_area(image_HW)
        img_mask = cv2.resize(self.png[:, :, 2], (image_HW[1], image_HW[0]))
        img_mask_ = img_mask[i1:i2, j1:j2]
        index = np.where(img_mask_ == 0)
        return index

    def get_mask_index(self, image_HW):
        i1, i2, j1, j2 = self.get_projection_area(image_HW)
        img_mask = cv2.resize(self.png[:, :, 2], (image_HW[1], image_HW[0]))
        img_mask_ = img_mask[i1:i2, j1:j2]
        index = np.where(img_mask_ > 0)
        return index

    

def set_config(path2work):
    
    screen_list = []
    num = 0
    
    while True:
        num += 1
        filename = path2work + 'screen_%d.png' % num
        if os.path.isfile(filename):
            scr = Screen(filename)
            screen_list.append(scr)
        else:
            if num == 1:
                raise Exception('%s is not exists.' % filename)
            break
    
    return screen_list
    



def add_equirectangular_margin(img, margin_x, margin_y=None):
    
    if margin_y == None:
        margin_y = margin_x
    if (type(margin_x) != int) or (type(margin_y) != int):
        print('The decimal point of margins are rounded up.')
        margin_x = int(np.ceil(margin_x))
        margin_y = int(np.ceil(margin_y))
    if img.ndim != 3:
        raise Exception('image dimention shoud be 3.')
    if (margin_y > img.shape[0]) or (margin_x > img.shape[1]):
        raise Exception('The size of the margin is up to the image size.')
    
    mx = margin_x
    my = margin_y
    H = img.shape[0] + margin_y * 2
    W = img.shape[1] + margin_x * 2
    D = img.shape[2]
    img_ = np.empty([H, W, D], dtype=img.dtype)
    img_[my:-my, mx:-mx, :] = img
    img_[:my, mx:W // 2 ,:] = img_[my:my * 2, W // 2:-mx, :][::-1]
    img_[:my, W // 2:-mx ,:] = img_[my:my * 2, mx:W // 2, :][::-1]
    img_[-my:, mx:W // 2, :] = img_[-my * 2:-my, W // 2:-mx, :][::-1]
    img_[-my:, W // 2:-mx, :] = img_[-my * 2:-my, mx:W // 2, :][::-1]
    img_[:, :mx, :] = img_[:, -mx * 2:-mx, :]
    img_[:, -mx:, :] = img_[:, mx:mx * 2, :]
    return img_



if __name__ == '__main__':
    pass
