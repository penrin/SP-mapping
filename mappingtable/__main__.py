import sys
import os
import argparse
import time

import util
import projector
import screen
import graycode






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
    
    

    # projector configuration
    proj_list = projector.set_config(path)
    projector.inspect_projectors(proj_list)

    # screen configuration
    screens = screen.set_config(path)
    
    # display & capture & analyse
    graycode.graycode_projection(proj_list, path)

    
    




