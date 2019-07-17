import numpy as np
import cv2
import json
from scipy import interpolate

import theta_s
import matplotlib.pyplot as plt

GRAY_VALUE = 186
KSIZE_MEDIAN_FILTER = 5 # 3, 5, 7, 9
KSIZE_SMOOTHING_X = 21 # odd
KSIZE_SMOOTHING_Y = 21 # odd
margin = KSIZE_SMOOTHING_Y // 2 + 1, KSIZE_SMOOTHING_X // 2 + 1


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
        config_sub['x_starting'] = int(proj.upperleft[1])
        config_sub['y_starting'] = int(proj.upperleft[0])
        
        
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
    

    
def add_equirectangular_margin(img, margin_x, margin_y=None):
    
    if margin_y == None:
        margin_y = margin_x
    if (type(margin_x) != int) or (type(margin_y) != int):
        print('The decimal point of margins are rounded up.')
        margin_x = int(np.ceil(margin_x))
        margin_y = int(np.ceil(margin_y))
    if img.ndim != 3:
        raise Exception('image dimention shoud be 3.')
    if (margin_y > img.shape[0]) or (margin_x > img.shape[1]):
        raise Exception('The size of the margin is up to the image size.')
    
    mx = margin_x
    my = margin_y
    H = img.shape[0] + my * 2
    W = img.shape[1] + mx * 2
    D = img.shape[2]
    img_ = np.empty([H, W, D], dtype=img.dtype)
    img_[my:-my, mx:-mx, :] = img
    img_[:my, mx:W // 2 ,:] = img_[my:my * 2, W // 2:-mx, :][::-1]
    img_[:my, W // 2:-mx ,:] = img_[my:my * 2, mx:W // 2, :][::-1]
    img_[-my:, mx:W // 2, :] = img_[-my * 2:-my, W // 2:-mx, :][::-1]
    img_[-my:, W // 2:-mx, :] = img_[-my * 2:-my, mx:W // 2, :][::-1]
    img_[:, :mx, :] = img_[:, -mx * 2:-mx, :]
    img_[:, -mx:, :] = img_[:, mx:mx * 2, :]
    return img_


def shift_horizontal(img, shift):
    if shift > 0:
        N = shift
    else:
        N = img.shape[1] + shift
    M = img.shape[1] - N
    img_ = np.empty_like(img)
    img_[:, :N] = img[:, -N:]
    img_[:, N:] = img[:, :M]
    return img_


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
        print('----- %d/%d -----' % (n + 1, N))

        proj_id = n + 1
        config_sub = config['parameters']['projector_%d' % proj_id]
        proj_HW = config_sub['y_num_pixel'], config_sub['x_num_pixel']
        x_starting = config_sub['x_starting']
        y_starting = config_sub['y_starting']

        
        scr = screen_list[n]
        i1, i2, j1, j2 = scr.get_evaluation_area_index()
        
        print('Decoding Gray-code pattern')
        # reference image
        filename = path + 'gray_proj%d_grey.jpg' % proj_id
        img_ref = imread(filename)
        img_ref = add_equirectangular_margin(img_ref, margin[1], margin[0])
        img_ref = shift_horizontal(img_ref, scr.horizontal_shift)
        img_ref = img_ref[i1:i2, j1:j2, :]

        # ----- x-axis -----
        BGR = config_sub['xgraycode_BGR']
        num_imgs = config_sub['xgraycode_num_image']
        nbits = config_sub['xgraycode_num_bits']
        offset = config_sub['xgraycode_offset']
        imgs_code = np.empty([i2 - i1, j2 - j1, nbits], dtype=np.bool)
        for i in range(num_imgs):
            # load image
            filename = path + 'gray_proj%d_x%d.jpg' % (proj_id, i)
            img = imread(filename)
            img = add_equirectangular_margin(img, margin[1], margin[0])
            img = shift_horizontal(img, scr.horizontal_shift)
            img = img[i1:i2, j1:j2, :]

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
        proj_x += x_starting - offset

        

        # ----- y-axis -----
        BGR = config_sub['ygraycode_BGR']
        num_imgs = config_sub['ygraycode_num_image']
        nbits = config_sub['ygraycode_num_bits']
        offset = config_sub['ygraycode_offset']
        imgs_code = np.empty([i2 - i1, j2 - j1, nbits], dtype=np.bool)
        for i in range(num_imgs):
            # load image
            filename = path + 'gray_proj%d_y%d.jpg' % (proj_id, i)
            img = imread(filename)
            img = add_equirectangular_margin(img, margin[1], margin[0])
            img = shift_horizontal(img, scr.horizontal_shift)
            img = img[i1:i2, j1:j2, :]

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
        proj_y += y_starting - offset
        
        # remove pulse noise from bit errors
        proj_x = cv2.medianBlur(proj_x, KSIZE_MEDIAN_FILTER)
        proj_y = cv2.medianBlur(proj_y, KSIZE_MEDIAN_FILTER)
        #plt.subplot(221)
        #plt.imshow(proj_x, cmap=plt.cm.prism)
        #plt.subplot(222)
        #plt.imshow(proj_y, cmap=plt.cm.prism)
        
        # smoothing
        proj_x = cv2.blur(proj_x, (KSIZE_SMOOTHING_X, KSIZE_SMOOTHING_Y))
        proj_y = cv2.blur(proj_y, (KSIZE_SMOOTHING_X, KSIZE_SMOOTHING_Y))
        #plt.subplot(223)
        #plt.imshow(proj_x[11:-11, 11:-11], cmap=plt.cm.prism)
        #plt.subplot(224)
        #plt.imshow(proj_y[11:-11, 11:-11], cmap=plt.cm.prism)
        #plt.show()
        #sys.exit()
        
        # pixel direction
        polar, azimuth = scr.get_direction_meshgrid()

        # estimate
        print('Estimating pixel direction')
        x1 = max(x_starting, int(np.floor(np.nanmin(proj_x))))
        x2 = min(int(np.ceil(np.nanmax(proj_x))), proj_HW[1] + x_starting)
        y1 = max(y_starting, int(np.floor(np.nanmin(proj_y))))
        y2 = min(int(np.ceil(np.nanmax(proj_y))), proj_HW[0] + y_starting)
        proj_y_interp, proj_x_interp = np.mgrid[y1:y2, x1:x2]
        
        index = scr.get_masked_index()
        azimuth_interp = interpolate.griddata(
                (proj_x[index], proj_y[index]), azimuth[index],
                (proj_x_interp, proj_y_interp), method='linear', fill_value=-1
                )
        polar_interp = interpolate.griddata(
                (proj_x[index], proj_y[index]), polar[index],
                (proj_x_interp, proj_y_interp), method='linear', fill_value=-1
                )
        
        plt.subplot(211)
        plt.imshow(azimuth_interp, cmap=plt.cm.prism)
        plt.subplot(212)
        plt.imshow(polar_interp, cmap=plt.cm.prism)
        plt.show()
        
         
        index = np.where(
                (scr.area_polar[0] <= polar_interp) &
                (polar_interp < scr.area_polar[1]) &
                (scr.area_azimuth[0] <= azimuth_interp) &
                (azimuth_interp < scr.area_azimuth[1])
                )
        proj_x_stack.append(proj_x_interp[index])
        proj_y_stack.append(proj_y_interp[index])
        azimuth_stack.append(azimuth_interp[index])
        polar_stack.append(polar_interp[index])
        
        
    print('---------------')
    print('save mapping table')
    proj_x_stack = np.hstack(proj_x_stack)
    proj_y_stack = np.hstack(proj_y_stack)
    azimuth_stack = np.hstack(azimuth_stack)
    polar_stack = np.hstack(polar_stack)
    filename = path + 'mapping_table.npz'
    np.savez(filename, y=proj_y_stack, x=proj_x_stack,
             azimuth=azimuth_stack, polar=polar_stack)


