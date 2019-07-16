import cv2
import numpy as np
from scipy import sparse




def make_convert_matrix(mapper, proj_HW, sph_HW, interp):

    if interp == 'bilinear':
        return mat_bilinear(mapper, proj_HW, sph_HW)
    elif interp == 'nearest':
        return mat_nearest(mapper, proj_HW, sph_HW)
    else:
        raise Exception('Not supported interpolation method.')


    
def mat_nearest(mapper, proj_HW, sph_HW):
    proj_x = mapper['x']
    proj_y = mapper['y'] 
    sph_x = mapper['azimuth'] * sph_HW[1] / 360
    sph_y = mapper['polar'] * sph_HW[0] / 180
    sx = np.round(sph_x) % sph_HW[1]
    sy = np.round(sph_y) % sph_HW[0]
    si = (sx + sph_HW[1] * sy).astype(np.int64)
    pi = proj_x + proj_HW[1] * proj_y
    out_size = proj_HW[0] * proj_HW[1]
    in_size = sph_HW[0] * sph_HW[1]
    mat = sparse.lil_matrix((out_size, in_size), dtype=np.float32)
    mat[pi, si] = 1.
    return mat


