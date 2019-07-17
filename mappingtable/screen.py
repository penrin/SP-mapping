import sys
import os
import cv2
import numpy as np
import graycode



class Screen():
    
    def __init__(self, filename, size, margin):
        
        png = cv2.imread(filename)
        if png is None:
            raise Exception('%s could not read.' % filename)
        
        png = self._resize_png(png, size)
        self.original_size = size
        png = self._add_margin(png, margin)
        self.margin = margin
        
        self.png, self.horizontal_shift = self._horizontal_centering_area(png)
        self.area_polar, self.area_azimuth = self._projection_area()
    
    
    def _resize_png(self, png, size):
        if png.shape[0:2] != size:
            png = cv2.resize(png, dsize=(size[1], size[0]))
        return png

    def _add_margin(self, png, margin):
        mx = margin[1]
        my = margin[0]
        H = png.shape[0] + my * 2
        W = png.shape[1] + mx * 2
        png_ = np.zeros([H, W, 3], dtype=png.dtype)
        png_[my:-my, mx:-mx] = png
        return png_
    

    def _horizontal_centering_area(self, img):
        blue = img[:, :, 0]
        green = img[:, :, 1]
        screen = np.logical_or(blue, green) 
        index_x = np.copy(np.where(screen)[1])
        x1, x2 = index_x.min(), index_x.max()
        W = screen.shape[1]
        if x1 == 0 and (x2 + 1) == W:
            index_x[np.where(index_x >= W / 2)[0]] -= W
        shift = int(W / 2 - np.mean(index_x))
        if shift > 0:
            N = shift
        else:
            N = img.shape[1] + shift
        M = img.shape[1] - N
        img_ = np.empty_like(img)
        img_[:, :N] = img[:, -N:]
        img_[:, N:] = img[:, :M]
        return img_, shift


    def _projection_area(self):
        '''
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
        '''
        i1, i2, j1, j2 = self.get_projection_area_index()
        H = self.original_size[0]
        W = self.original_size[1]
        polar1 = (i1 - self.margin[0]) * 180 / H
        polar2 = (i2 - self.margin[0]) * 180 / H
        azimuth1 = (j1 - self.margin[1] - self.horizontal_shift) * 360 / W
        azimuth2 = (j2 + self.margin[1] - self.horizontal_shift) * 360 / W
        return (polar1, polar2), (azimuth1, azimuth2)
    
    
    def get_projection_area_index(self):
        blue = self.png[:, :, 0]
        green = self.png[:, :, 1]
        screen = np.logical_or(blue, green) 
        ij_screen = np.where(screen)
        i1, i2 = ij_screen[0].min(), ij_screen[0].max() + 1
        j1, j2 = ij_screen[1].min(), ij_screen[1].max() + 1
        return i1, i2, j1, j2
    
    
    def get_evaluation_area_index(self):
        i1, i2, j1, j2 = self.get_projection_area_index()
        i1 -= self.margin[0]
        i2 += self.margin[0]
        j1 -= self.margin[1]
        j2 += self.margin[1]
        return i1, i2, j1, j2


    def get_direction_meshgrid(self):
        i1, i2, j1, j2 = self.get_projection_area_index()
        y = np.arange(i1, i2) - self.margin[0]
        x = np.arange(j1, j2) - self.margin[1] - self.horizontal_shift
        H = self.original_size[0]
        W = self.original_size[1]
        azimuth, polar = np.meshgrid(x * 360 / W, y * 180 / H)
        return polar, azimuth


    def get_masked_index(self):
        i1, i2, j1, j2 = self.get_projection_area_index()
        img_mask = self.png[:, :, 2][i1:i2, j1:j2]
        index = np.where(img_mask == 0)
        return index


    def get_mask_index(self):
        i1, i2, j1, j2 = self.get_projection_area()
        img_mask = self.png[:, :, 2][i1:i2, j1:j2]
        index = np.where(img_mask > 0)
        return index



def set_config(path2work):
    
    img = cv2.imread(path2work + 'gray_proj1_grey.jpg')
    if img is None:
        raise Exception('%s could not read.' % filename)
    size = img.shape[0:2]
    margin = graycode.margin
    print('margin:', margin)
    

    screen_list = []
    num = 0
    
    while True:
        num += 1
        filename = path2work + 'screen_%d.png' % num
        if os.path.isfile(filename):
            scr = Screen(filename, size, margin)
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
    H = img.shape[0] + my * 2
    W = img.shape[1] + mx * 2
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


def shift_horizontal(img, shift):
    if shift > 0:
        N = shift
    else:
        N = img.shape[1] + shift
    M = img.shape[1] - N
    img_ = np.empty_like(img)
    img_[:, :N] = img[:, -N:]
    img_[:, N:] = img[:, :M]
    return img_



if __name__ == '__main__':

    path = '../../workfolder'
    if path[-1] != '/':
        path += '/'
    screen_list = set_config(path)
    import matplotlib.pyplot as plt
    plt.imshow(screen_list[2].png[:, :, ::-1])
    plt.show()
    
