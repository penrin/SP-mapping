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
    x, y, polar, azimuth, ovlp_x, ovlp_y, ovlp_w,\
                        tone_input, tone_output, proj_HW = mapper


    
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
    
    plt.subplots(figsize=(12, 12))
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
    
    print('---------------------------')
    print('saving mapping table...', end='')
    p = util.Propeller()
    p.start()

    x, y, polar, azimuth, ovlp_x, ovlp_y, ovlp_weight,\
                        tone_input, tone_output, proj_HW = mapper
    
    filename = path + 'mapping_table.npz'
    np.savez(filename, 
            y = y.astype(np.int64),
            x = x.astype(np.int64),
            polar = polar.astype(np.float64),
            azimuth = azimuth.astype(np.float64),
            ovlp_x = ovlp_x.astype(np.int64),
            ovlp_y = ovlp_y.astype(np.int64),
            ovlp_weight = ovlp_weight.astype(np.float64),
            tone_input = tone_input.astype(np.float64),
            tone_output = tone_output.astype(np.float64),
            proj_HW = proj_HW.astype(np.int64)
            )

    p.end()





if __name__ == '__main__':
     
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to working folder')
    parser.add_argument('-c', '--capture', action='store_true', help='capturing only')
    parser.add_argument('-a', '--analysis', action='store_true', help='analysis only')
    parser.add_argument('--ev', type=float, default=0.0, help='RICOH THETA exposure')
    parser.add_argument('--grey', type=int, default=186, help='Brightness value of reference grey 0--255')
    #parser.add_argument('--rgb', action='store_true', help='using RGB pattern')
    #parser.add_argument('--pn', action='store_true', help='using negative pattern')
    parser.add_argument('--monocolor', action='store_true', help='using mono color pattern')
    parser.add_argument('--posi', action='store_true', help='use only positive pattern (not recommended)')
    parser.add_argument('--dryrun', action='store_true', help='dry run (do not connect to the THETA)')
    parser.add_argument('--savepattern', action='store_true', help='save the projection patterns')
    args = parser.parse_args()
    path = args.path
    cap = args.capture
    ana = args.analysis
    EV = args.ev
    GREY_VALUE = args.grey
    BGR = not args.monocolor
    PN = not args.posi
    DRY = args.dryrun
    save_pattern = args.savepattern

    
    if (cap == False) & (ana == False):
        parser.print_help()
    
    if path[-1] != '/':
        path += '/'
    if not os.path.isdir(path):
        raise Exception('%s is not exists.' % path)

    # gray-code pattern display & capture
    if cap == True:
        proj_list = projector.set_config(path)
        projector.inspect_projectors(proj_list)
        if os.name == 'nt':
            print('Windows mode')
            TK = True
        else:
            TK = False
        graycode.graycode_projection(
                proj_list, path,
                save_pattern=save_pattern, EV=EV,
                GREY_VALUE=GREY_VALUE, BGR=BGR,
                PN=PN, TK=TK, DRY=DRY
                )

    # gray-code pattern analysis
    if ana == True:
        screen_list = screen.set_config(path)
        mapper = graycode.graycode_analysis(screen_list, path)
        check_mapper(mapper)
        save_mapper(mapper)
    


    
