import sys
import os
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import util
import projector
import screen
import graycode




def check_mapper(mapper):
    x, y, polar, azimuth, ovlp_x, ovlp_y, ovlp_w = mapper
    
    img = cv2.imread(path + 'projector_1.png')
    H, W, _ = img.shape
    
    img_x = np.zeros([H, W])
    img_y = np.zeros([H, W])
    pi = x + W * y
    
    img_x.reshape(-1)[pi] = polar
    img_y.reshape(-1)[pi] = azimuth
    
    img_ovlp = np.zeros([H, W])
    ovlp_pi = ovlp_x + W * ovlp_y
    img_ovlp.reshape(-1)[pi] = 1
    img_ovlp.reshape(-1)[ovlp_pi] = ovlp_w

    img_x[np.where(img_x == 0)] = np.nan
    img_y[np.where(img_y == 0)] = np.nan
    img_ovlp[np.where(img_ovlp == 0)] = np.nan
    
    plt.subplots(figsize=(6, 12))
    plt.subplot(311)
    plt.imshow(img_x, cmap=plt.cm.prism)
    plt.subplot(312)
    plt.imshow(img_y, cmap=plt.cm.prism)
    plt.subplot(313)
    plt.imshow(img_ovlp, cmap=plt.cm.jet)
    plt.tight_layout()
    plt.savefig(path + 'plt_mapper.pdf')
    plt.close()




def save_mapper(mapper):
    x, y, polar, azimuth, ovlp_x, ovlp_y, ovlp_weight = mapper
    
    filename = path + 'mapping_table.npz'
    np.savez(filename, 
            y=y, x=x,
            polar=polar,
            azimuth=azimuth,
            ovlp_x=ovlp_x,
            ovlp_y=ovlp_y,
            ovlp_weight=ovlp_weight
            )

    




if __name__ == '__main__':
     

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1', help='path to working folder')
    args = parser.parse_args()
    
    path = args.arg1
    '''
    path = '../../workfolder'
    if path[-1] != '/':
        path += '/'
    if not os.path.isdir(path):
        raise Exception('%s is not exists.' % path)


    '''
    # gray-code pattern display & capture
    proj_list = projector.set_config(path)
    projector.inspect_projectors(proj_list)
    graycode.graycode_projection(proj_list, path, save_pattern=False)
    '''
    
    # gray-code pattern analysis
    screen_list = screen.set_config(path)
    mapper = graycode.graycode_analysis(screen_list, path)
    
    
    print('---------------')
    print('save mapping table')
    check_mapper(mapper)
    save_mapper(mapper)
    


    
