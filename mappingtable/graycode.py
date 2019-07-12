import numpy as np
import cv2
import json

import theta_s


GRAY_VALUE = 186


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
    config['parameters'] = {}


    # THETA
    theta = theta_s.ThetaS()
    
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
    
    cv2.close()
    
    # save config
    filename = path + 'graycode_config.json'
    f = open(filename, 'w')
    json.dump(config, f, indent=4)
    f.close()
    
    # save images
    for i in range(len(filename_list)):
        theta.save(URI_list[i], path + filename_list[i])


