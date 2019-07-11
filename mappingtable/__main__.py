import sys
import os
import argparse
import cv2
import time

import util
import projector
import screen
import theta_s
import graycode



def graycode_projection(proj_list, path):

    # THETA
    theta = theta_s.ThetaS()
    
    # display and caputure
    
    URI_list = []
    filename_list = []
    
    for n, proj in enumerate(proj_list):
        
        # grey color
        grey = proj.gen_canvas()
        grey[:] = GRAY_VALUE
        disp_img = proj.add_base(grey)
        cv2.imshow('SPM', disp_img)
        
        if n == 0:
            print('Ready. Press any key to start.')
            cv2.waitKey(0)
            print('--------------------------------')
            print('Start measurement. Don\'t touch.')
            print('--------------------------------')
        else:
            cv2.waitKey(10)
                
        # adjust exposure
        print('Adjusting exposure...')
        filename = path + 'auto_proj%d.jpg' % (n + 1)
        iso, shutter = theta.auto_adjust_exposure(filename)
        print('ISO:', iso, ', Shutter:', shutter)
        
        
        URI = theta.take()
        URI_list.append(URI)
        filename = 'gray_proj%d_grey.jpg' % (n + 1)
        filename_list.append(filename)

        # x-axis
        imgs, nbits = graycode.graycodepattern(proj.aspect, axis='x', BGR=True)
        for i in range(len(imgs)):
            # display
            disp_img = proj.add_base(imgs[i])
            cv2.imshow('SPM', disp_img)
            cv2.waitKey(10)
            time.sleep(0.1)
            # capture
            URI = theta.take()
            URI_list.append(URI)
            filename = 'gray_proj%d_x%d.jpg' % (n + 1, i)
            filename_list.append(filename)
            
        # y-axis
        imgs, nbits = graycode.graycodepattern(proj.aspect, axis='y', BGR=True)
        for i in range(len(imgs)):
            # display
            disp_img = proj.add_base(imgs[i])
            cv2.imshow('SPM', disp_img)
            cv2.waitKey(10)
            time.sleep(0.1)
            # capture
            URI = theta.take()
            URI_list.append(URI)
            filename = 'gray_proj%d_y%d.jpg' % (n + 1, i)
            filename_list.append(filename)
    
    
    # save images
    for i in range(len(filename_list)):
        theta.save(URI_list[i], path + filename_list[i])
        
    





if __name__ == '__main__':
    
   
    cv2.namedWindow('SPM', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('SPM', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    GRAY_VALUE = 119
        

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
    
    

    # projector configuration
    proj_list = projector.set_config(path)
    projector.inspect_projectors(proj_list)

    # screen configuration
    screens = screen.set_config(path)
    
    # display & capture
    graycode_projection(proj_list, path)

    # analyse
    
    




