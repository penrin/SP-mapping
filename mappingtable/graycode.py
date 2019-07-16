import numpy as np
import cv2
import json
from scipy import interpolate

import theta_s
import matplotlib.pyplot as plt

GRAY_VALUE = 186
KSIZE_MEDIAN_FILTER = 5
KSIZE_SMOOTHING = 20 # moving average

# graycode encoding
def gray_encode(n:int) -> int:
    gray = n ^ (n >> 1)
    return gray

# graycode decoding
def gray_decode(gray:int) -> int:
    n = gray
    i = 1
    while (gray >> i) > 0:
        n = n ^ (gray >> i)
        i += 1
    return n

# gray code pattern
def int2binarray(n:int, nbits:int):
    arr = np.zeros(nbits, dtype=np.uint8)
    b = bin(n)
    i = 1
    while (b[-i] != 'b') and (i <= nbits):
        if b[-i] == '1':
            arr[-i] = 1
        i += 1
    return arr




def graycodepattern(aspect, axis='x', BGR=False):
    
    if axis == 'x':
        img, nbits, offset = _graycodepattern_x(aspect)
    elif axis == 'y':
        img, nbits, offset = _graycodepattern_y(aspect)
    else:
        raise Exception('axis must be x or y.')
    
    imgs = []
    if BGR == False:
        for i in range(nbits):
            imgs.append(img[:, :, i])

    if BGR == True:
        n_img = int(np.ceil(nbits / 3))
        img_bgr = np.zeros([aspect[0], aspect[1], n_img, 3], dtype=np.uint8)
        
        for i in range(n_img):
            img_bgr = np.zeros([aspect[0], aspect[1], 3], dtype=np.uint8)
            for j in range(3):
                if (i * 3 + j) >= nbits:
                    break
                img_bgr[:, :, j] = img[:, :, i * 3 + j]
            imgs.append(img_bgr)

    return imgs, nbits, int(offset)



def _graycodepattern_x(aspect):
    nbits = int(np.ceil(np.log2(aspect[1])))
    x = np.arange(aspect[1])
    offset_x = (2 ** nbits - aspect[1]) // 2
    gray_x = gray_encode(x + offset_x)
    img = np.empty([aspect[0], aspect[1], nbits], dtype=np.uint8)
    for i in range(len(gray_x)):
        img[:, i, :] = int2binarray(gray_x[i], nbits).reshape(1, -1)
    return img * 255, nbits, offset_x


def _graycodepattern_y(aspect):
    nbits = int(np.ceil(np.log2(aspect[0])))
    y = np.arange(aspect[0])
    offset_y = (2 ** nbits - aspect[0]) // 2
    gray_y = gray_encode(y + offset_y)
    img = np.empty([aspect[0], aspect[1], nbits], dtype=np.uint8)
    for i in range(len(gray_y)):
        img[i, :, :] = int2binarray(gray_y[i], nbits).reshape(1, -1)
    return img * 255, nbits, offset_y





def graycode_projection(proj_list, path):
    
    cv2.namedWindow('SPM', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('SPM', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    config = {}
    config['num_projectors'] = len(proj_list)
    config['camera'] = {}
    config['parameters'] = {}


    # THETA
    theta = theta_s.ThetaS()
    config['camera']['model'] = 'RICOH THETA S'
    HW = theta.get_imageSize()
    config['camera']['height'] = HW[0]
    config['camera']['width'] = HW[1]
    
    # display and caputure
    URI_list = []
    filename_list = []

    for n, proj in enumerate(proj_list):
        
        config_sub = {}
        config_sub['x_num_pixel'] = int(proj.aspect[1])
        config_sub['y_num_pixel'] = int(proj.aspect[0])

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
        ret = graycodepattern(proj.aspect, axis='x', BGR=True)
        imgs, nbits, offset = ret
        for i in range(len(imgs)):
            # display
            disp_img = proj.add_base(imgs[i])
            cv2.imshow('SPM', disp_img)
            cv2.waitKey(10)
            # capture
            URI = theta.take()
            URI_list.append(URI)
            filename = 'gray_proj%d_x%d.jpg' % (n + 1, i)
            filename_list.append(filename)
        config_sub['xgraycode_BGR'] = True
        config_sub['xgraycode_num_image'] = len(imgs)
        config_sub['xgraycode_num_bits'] = nbits
        config_sub['xgraycode_offset'] = offset

        # y-axis
        ret = graycodepattern(proj.aspect, axis='y', BGR=True)
        imgs, nbits, offset = ret
        for i in range(len(imgs)):
            # display
            disp_img = proj.add_base(imgs[i])
            cv2.imshow('SPM', disp_img)
            cv2.waitKey(10)
            # capture
            URI = theta.take()
            URI_list.append(URI)
            filename = 'gray_proj%d_y%d.jpg' % (n + 1, i)
            filename_list.append(filename)
        config_sub['ygraycode_BGR'] = True
        config_sub['ygraycode_num_image'] = len(imgs)
        config_sub['ygraycode_num_bits'] = nbits
        config_sub['ygraycode_offset'] = offset
        
        config['parameters']['projector_%d' % (n + 1)] = config_sub
     
    # save config
    filename = path + 'graycode_config.json'
    f = open(filename, 'w')
    json.dump(config, f, indent=4)
    f.close()
    
    # save images
    print('save images')
    for i in range(len(filename_list)):
        theta.save(URI_list[i], path + filename_list[i])




def imread(filename):
    img = cv2.imread(filename)
    if img is None:
        raise Exception('Cannot read %d' % filename)
    else:
        return img
    

def graycode_analysis(screen_list, path):
    
    # load configuration
    filename = path + 'graycode_config.json'
    f = open(filename, 'r')
    config = json.load(f)
    
    img_HW = config['camera']['height'], config['camera']['width']
    
    N = config['num_projectors']
    proj_x_stack, proj_y_stack = [], []
    azimuth_stack, polar_stack = [], []
    for n in range(N):

        proj_id = n + 1
        config_sub = config['parameters']['projector_%d' % proj_id]
        proj_HW = config_sub['y_num_pixel'], config_sub['x_num_pixel']
        screen = screen_list[n]
        
        i1, i2, j1, j2 = screen.get_projection_area(img_HW)
        # reference image
        filename = path + 'gray_proj%d_grey.jpg' % proj_id
        img_ref = imread(filename)[i1:i2, j1:j2, :]
        
        # ----- x-axis -----
        BGR = config_sub['xgraycode_BGR']
        num_imgs = config_sub['xgraycode_num_image']
        nbits = config_sub['xgraycode_num_bits']
        offset = config_sub['xgraycode_offset']
        imgs_code = np.empty([i2 - i1, j2 - j1, nbits], dtype=np.bool)
        for i in range(num_imgs):
            # load image
            filename = path + 'gray_proj%d_x%d.jpg' % (proj_id, i)
            img = imread(filename)[i1:i2, j1:j2, :]
            # judge 0 or 1
            if BGR:
                code = (img > img_ref)
                for j in range(3):
                    if (3 * i + j) >= nbits: break
                    imgs_code[:, :, 3 * i + j] = code[:, :, j]
            else:
                code = (img[:, :, 1] > img_ref[:, :, 1]) # green layer
                imgs_code[:, :, i] = code
        # decode
        imgs_bin = np.empty_like(imgs_code, dtype=np.bool)
        imgs_bin[:, :, 0] = imgs_code[:, :, 0]
        for i in range(1, nbits):
            imgs_bin[:, :, i] = np.logical_xor(
                                 imgs_bin[:, :, i - 1], imgs_code[:, :, i])
        weight = 2 ** np.arange(nbits)[::-1].reshape(1, 1, -1)
        proj_x = np.sum(imgs_bin * weight, axis=-1).astype(np.float32)
        proj_x -= offset
        

        # ----- y-axis -----
        BGR = config_sub['ygraycode_BGR']
        num_imgs = config_sub['ygraycode_num_image']
        nbits = config_sub['ygraycode_num_bits']
        offset = config_sub['ygraycode_offset']
        imgs_code = np.empty([i2 - i1, j2 - j1, nbits], dtype=np.bool)
        for i in range(num_imgs):
            # load image
            filename = path + 'gray_proj%d_y%d.jpg' % (proj_id, i)
            img = imread(filename)[i1:i2, j1:j2, :]
            # judge 0 or 1
            if BGR:
                code = (img > img_ref)
                for j in range(3):
                    if (3 * i + j) >= nbits: break
                    imgs_code[:, :, 3 * i + j] = code[:, :, j]
            else:
                code = (img[:, :, 1] > img_ref[:, :, 1]) # green layer
                imgs_code[:, :, i] = code
        # decode
        imgs_bin = np.empty_like(imgs_code, dtype=np.bool)
        imgs_bin[:, :, 0] = imgs_code[:, :, 0]
        for i in range(1, nbits):
            imgs_bin[:, :, i] = np.logical_xor(
                                 imgs_bin[:, :, i - 1], imgs_code[:, :, i])
        weight = 2 ** np.arange(nbits)[::-1].reshape(1, 1, -1)
        proj_y = np.sum(imgs_bin * weight, axis=-1).astype(np.float32)
        proj_y -= offset
        
        # remove pulse noise from bit errors
        proj_x = cv2.medianBlur(proj_x, KSIZE_MEDIAN_FILTER)
        proj_y = cv2.medianBlur(proj_y, KSIZE_MEDIAN_FILTER)
        
        # smoothing
        proj_x = cv2.blur(proj_x, (KSIZE_SMOOTHING, KSIZE_SMOOTHING))
        proj_y = cv2.blur(proj_y, (KSIZE_SMOOTHING, KSIZE_SMOOTHING))

        # pixel direction
        azimuth, polar = np.meshgrid(
                np.arange(j1, j2) * 360 / img_HW[1],
                np.arange(i1, i2) * 180 / img_HW[0],
                )

        # estimate
        x1 = max(0, int(np.floor(np.nanmin(proj_x))))
        x2 = min(int(np.ceil(np.nanmax(proj_x))), proj_HW[1])
        y1 = max(0, int(np.floor(np.nanmin(proj_y))))
        y2 = min(int(np.ceil(np.nanmax(proj_y))), proj_HW[0])
        proj_y_interp, proj_x_interp = np.mgrid[y1:y2, x1:x2]
        
        index = screen.get_masked_index(img_HW)
        azimuth_interp = interpolate.griddata(
                (proj_x[index], proj_y[index]), azimuth[index],
                (proj_x_interp, proj_y_interp), method='linear', fill_value=-1
                )
        polar_interp = interpolate.griddata(
                (proj_x[index], proj_y[index]), polar[index],
                (proj_x_interp, proj_y_interp), method='linear', fill_value=-1
                )

        index = np.where(
                (screen.area_polar[0] <= polar_interp) &
                (polar_interp < screen.area_polar[1]) &
                (screen.area_azimuth[0] <= azimuth_interp) &
                (azimuth_interp < screen.area_azimuth[1])
                )
        proj_x_stack.append(proj_x_interp[index])
        proj_y_stack.append(proj_y_interp[index])
        azimuth_stack.append(azimuth_interp[index])
        polar_stack.append(polar_interp[index])
        
    
    proj_x_stack = np.hstack(proj_x_stack)
    proj_y_stack = np.hstack(proj_y_stack)
    azimuth_stack = np.hstack(azimuth_stack)
    polar_stack = np.hstack(polar_stack)
    filename = path + 'mapping_table.npz'
    np.savez(filename, y=proj_y_stack, x=proj_x_stack,
             azimuth=azimuth_stack, polar=polar_stack)
    
