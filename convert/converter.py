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
    mat = sparse.lil_matrix(
            (proj_HW[0] * proj_HW[1], sph_HW[0] * sph_HW[1]), dtype=np.float32)
    mat[pi, si] = 1.
    return mat


def mat_bilinear(mapper, proj_HW, sph_HW):
    proj_x = mapper['x']
    proj_y = mapper['y']
    sph_x = mapper['azimuth'] * sph_HW[1] / 360
    sph_y = mapper['polar'] * sph_HW[0] / 180
    
    sx1 = np.floor(sph_x) + 1
    sx2 = np.floor(sph_x)
    sy1 = np.floor(sph_y) + 1
    sy2 = np.floor(sph_y)
    sx1d = sx1 - sph_x
    sx2d = sph_x - sx2
    sy1d = sy1 - sph_y
    sy2d = sph_y - sy2
    
    sx1 %= sph_HW[1]
    sx2 %= sph_HW[1]
    sy1 %= sph_HW[0]
    sy2 %= sph_HW[0]
    
    pi = proj_x + proj_HW[1] * proj_y
    si1 = (sx1 + sph_HW[1] * sy1).astype(np.uint64)
    si2 = (sx1 + sph_HW[1] * sy2).astype(np.uint64)
    si3 = (sx2 + sph_HW[1] * sy1).astype(np.uint64)
    si4 = (sx2 + sph_HW[1] * sy2).astype(np.uint64)
    
    mat = sparse.lil_matrix(
            (proj_HW[0] * proj_HW[1], sph_HW[0] * sph_HW[1]), dtype=np.float32)
    mat[pi, si1] = sx2d * sy2d
    mat[pi, si2] = sx2d * sy1d
    mat[pi, si3] = sx1d * sy2d
    mat[pi, si4] = sx1d * sy1d
    return mat

    

