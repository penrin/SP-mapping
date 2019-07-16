import sys
import cv2
import numpy as np
from scipy import sparse

import converter




def convert_image():
    sph_HW = img.shape[0], img.shape[1]
    print('Making convert matrix')
    mat = converter.make_convert_matrix(mapper, proj_HW, sph_HW, interp)
    mat = sparse.csr_matrix(mat)
    print('Convert')
    img_linear = (img / 255) ** gamma
    out = np.empty([proj_HW[0], proj_HW[1], 3], dtype=np.float32)
    for i in range(3):
        b = img_linear[:, :, i].reshape(-1, 1)
        out[:, :, i] = mat.dot(b).reshape(proj_HW)
    out = 255 * out ** (1 / gamma)
    out[np.where(out < 0)] = 0
    out[np.where(out > 255)] = 255
    cv2.imwrite(outfilename, out.astype(np.uint8))
    return


def convert_video(cap, mapper, filename, options):
    pass





def check_mapper():
    
    proj_x = mapper['x']
    proj_y = mapper['y'] 
    sph_x = mapper['azimuth']
    sph_y = mapper['polar']
    img_x = np.zeros(proj_HW)
    img_y = np.zeros(proj_HW)
    pi = proj_x + proj_HW[1] * proj_y
    img_x.reshape(-1)[pi] = sph_x
    img_y.reshape(-1)[pi] = sph_y

    import matplotlib.pyplot as plt
    plt.subplot(211)
    plt.imshow(img_x)
    plt.subplot(212)
    plt.imshow(img_y)
    plt.show()

    


if __name__ == '__main__':
    
    '''
    parser = argparse.ArgumentParser()
    #parser.add_argument('arg1', help='path to working folder')
    parser.add_argument('-i', type=str, help='input image or movie filename', required=True)
    parser.add_argument('-d', type=str, help='working directory', required=True)
    parser.add_argument('--interp', type=str, default='bilinear')
    parser.add_argument('--gamma', type=float, default=2.2, help='Gamma')
    parser.add_argument('arg1', default=None)
    args = parser.parse_args()
    path = args.d
    infilename = args.i
    outfilename = args.arg1
    '''

    
    path = '../../workfolder'
    if path[-1] != '/':
        path += '/'
    
    mapper = np.load(path + 'mapping_table.npz')    

    infilename = '/Users/penrin/Desktop/workfolder/sample_0.png'
    outfilename = path + 'output.jpg'
    
    gamma = 1
    interp = 'bilinear'
    proj_img = cv2.imread(path + 'projector_1.png')
    proj_HW = proj_img.shape[0], proj_img.shape[1]
    
    #check_mapper()
    #sys.exit()


    img = cv2.imread(infilename)
    if img is not None:
        convert_image()
    else:
        cap = cv2.VideoCapture(infilename)
        convert_video()
        
