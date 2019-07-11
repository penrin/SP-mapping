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




cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('gray', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


GRAY_VALUE = 119

def main():
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
        
    # THETA
    #theta = theta_s.ThetaS()
    
    
    
    
    # display and caputure
    for n, proj in enumerate(proj_list):
        
        # grey color
        grey = proj.gen_canvas()
        grey[:] = GRAY_VALUE
        disp_img = proj.add_base(grey)
        cv2.imshow('gray', disp_img)
        
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
        #theta.adjust_exposure()
        time.sleep(0.1)
        
        # graycode pattern
        imgs, nbits = graycode.graycodepattern(proj.aspect, axis='x', BGR=True)
        for i in range(len(imgs)):
            disp_img = proj.add_base(imgs[i])
            cv2.imshow('gray', disp_img)
            cv2.waitKey(10)
            time.sleep(0.1)
            
        imgs, nbits = graycode.graycodepattern(proj.aspect, axis='y', BGR=True)
        for i in range(len(imgs)):
            disp_img = proj.add_base(imgs[i])
            cv2.imshow('gray', disp_img)
            cv2.waitKey(10)
            time.sleep(0.1)
        
        # graycode
        




if __name__ == '__main__':
    

    main()
