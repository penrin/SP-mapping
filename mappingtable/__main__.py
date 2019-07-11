import sys
import os
import argparse

import util
import projector
import theta_s



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
        print(path, 'is not exists.')
        sys.exit()
    
    
    

    # projector configutation
    proj_integ = projector.set_config(path)
    
    
    # make display pattern
    



    # display and caputure
    



    # 
    





if __name__ == '__main__':
    

    main()
