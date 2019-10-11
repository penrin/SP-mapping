import sys
import os
import cv2
import numpy as np
import json
import graycode



class Screen():
    
    def __init__(self, filename, size, overlap_angle, tone_curve):
        
        png = cv2.imread(filename)
        if png is None:
            msg = '%s could not read.' % filename
            raise Exception(msg)
        
        png = self._resize_png(png, size)
        png_centering, self.horizontal_shift = self._horizontal_centering_area(png)
        self.horizontal_shift_deg = self.horizontal_shift / size[1] * 360

        self.png = self._add_margin(png_centering, graycode.margin)
        self.original_size = size
        self.margin = graycode.margin
        self.overlap_angle = overlap_angle
        self.tone_input = tone_curve[0, :]
        self.tone_output = tone_curve[1, :]
        
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
        green = img[:, :, 1]
        screen = (green > 0)
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
        green = self.png[:, :, 1]
        screen = (green > 0)
        ij_screen = np.where(screen)
        i1, i2 = ij_screen[0].min(), ij_screen[0].max() + 1
        j1, j2 = ij_screen[1].min(), ij_screen[1].max() + 1
        H = self.original_size[0]
        W = self.original_size[1]
        polar1 = (i1 - self.margin[0]) * 180 / H
        polar2 = (i2 - self.margin[0]) * 180 / H
        azimuth1 = (j1 - self.margin[1]) * 360 / W
        azimuth2 = (j2 - self.margin[1]) * 360 / W
        azimuth1 -= self.overlap_angle / 2
        azimuth2 += self.overlap_angle / 2
        return (polar1, polar2), (azimuth1, azimuth2)


     
    def get_projection_area_index(self):
        green = self.png[:, :, 1]
        screen = (green > 0)
        ij_screen = np.where(screen)
        i1, i2 = ij_screen[0].min(), ij_screen[0].max() + 1
        j1, j2 = ij_screen[1].min(), ij_screen[1].max() + 1
        W = self.original_size[1]
        j1 -= int(self.overlap_angle / 2 * W / 360)
        j2 += int(np.ceil(self.overlap_angle / 2 * W / 360))
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
        x = np.arange(j1, j2) - self.margin[1]
        H = self.original_size[0]
        W = self.original_size[1]
        azimuth, polar = np.meshgrid(x * 360 / W, y * 180 / H)
        return polar, azimuth


    def get_masked_index(self):
        i1, i2, j1, j2 = self.get_projection_area_index()
        img_mask = self.png[:, :, 2][i1:i2, j1:j2]
        index = np.where(img_mask == 0)
        return index






def set_config(path2work):
    
    # get THETA S image size 
    img = cv2.imread(path2work + 'gray_proj1_x0_posi.jpg')
    if img is None:
        msg = '%s could not read.' % (path2work + 'gray_proj1_x0_posi.jpg')
        raise Exception(msg)
    size = img.shape[0:2]
    
    
    # overlap setting
    filename = path2work + 'screen_overlap.json'
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            overlap_setting = json.load(f)
        overlap_angle = overlap_setting['overlap_angle']
        tone_curve = np.array([
            overlap_setting['projector_tone_curve']['input'],
            overlap_setting['projector_tone_curve']['output']
            ])
    else:
        overlap_angle = 0
        tone_curve = np.array([[], []])

    
    screen_list = []
    num = 0
    
    while True:
        num += 1
        filename = path2work + 'screen_%d.png' % num
        if os.path.isfile(filename):
            scr = Screen(filename, size, overlap_angle, tone_curve)
            screen_list.append(scr)
        else:
            if num == 1:
                raise Exception('%s is not exists.' % filename)
            break
    
    return screen_list
    



if __name__ == '__main__':

    path = '../../workfolder'
    if path[-1] != '/':
        path += '/'
    screen_list = set_config(path)
    import matplotlib.pyplot as plt
    plt.imshow(screen_list[2].png[:, :, ::-1])
    plt.show()
    
