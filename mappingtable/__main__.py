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



    # gray-code pattern display & capture
    proj_list = projector.set_config(path)
    projector.inspect_projectors(proj_list)
    #graycode.graycode_projection(proj_list, path)
    

    # gray-code pattern analysis
    screen_list = screen.set_config(path)
    graycode.graycode_analysis(screen_list, path)
    
    




