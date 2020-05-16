import numpy as np
import cv2
import json
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import tkinter
from PIL import Image, ImageTk
import time

import theta_s
import concavehull
import util


GREY_VALUE = 100
EV = -2


KSIZE_MEDIAN_FILTER = 5 # 3, 5, 7, 9
KSIZE_SMOOTHING_X = 21 # odd
KSIZE_SMOOTHING_Y = 21 # odd
margin = KSIZE_SMOOTHING_Y // 2 + 1, KSIZE_SMOOTHING_X // 2 + 1

CV_DISP_NAME = 'SP-mapping'

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




def graycodepattern(aspect, axis='x', BGR=False, reverse=False):
    
    if axis == 'x':
        img, nbits, offset = _graycodepattern_x(aspect)
    elif axis == 'y':
        img, nbits, offset = _graycodepattern_y(aspect)
    else:
        raise Exception('axis must be x or y.')
    
    imgs = []
    if BGR == False:
        for i in range(nbits):
            img_bgr = np.zeros([aspect[0], aspect[1], 3], dtype=np.uint8)
            if reverse == False:
                img_bgr[:, :, 1] = img[:, :, i]
            else:
                img_bgr[:, :, 1] = 255 - img[:, :, i]
            imgs.append(img_bgr)

    if BGR == True:
        n_img = int(np.ceil(nbits / 3))
        for i in range(n_img):
            img_bgr = np.zeros([aspect[0], aspect[1], 3], dtype=np.uint8)
            for j in range(3):
                if (i * 3 + j) >= nbits:
                    break
                if reverse == False:            
                    img_bgr[:, :, j] = img[:, :, i * 3 + j]
                else:
                    img_bgr[:, :, j] = 255 - img[:, :, i * 3 + j]
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



def tk_imshow(canvas, img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tk  = ImageTk.PhotoImage(image_pil)
    canvas.create_image(0, 0, image=image_tk, anchor='nw', tag='img')
    canvas.pack()
    canvas.update()
    canvas.delete('img')
    return

def cv_imshow(img):
    cv2.imshow(CV_DISP_NAME, img)
    cv2.waitKey(10)
    return


def graycode_projection(proj_list, path, save_pattern=False,
            EV=0, GREY_VALUE=100, BGR=False, PN=True, TK=False):
    
    config = {}
    config['num_projectors'] = len(proj_list)
    config['projector_whole_HW'] = proj_list[0].base_aspect
    config['camera'] = {}
    config['parameters'] = {}
    
    # THETA
    theta = theta_s.ThetaS()
    theta.set_stillmode()
    config['camera']['model'] = theta.get_modelname()
    HW = theta.get_imageSize()
    config['camera']['height'] = HW[0]
    config['camera']['width'] = HW[1]
    
    # display and caputure
    URI_list = []
    filename_list = []
    
    if TK == True:
        tkroot = tkinter.Tk()
        tkroot.attributes("-fullscreen", True)
        
        print('Ready. Press enter key to start.')
        input()
        tkroot.focus_force()
        tkcanvas = tkinter.Canvas(tkroot, width=proj_list[0].base_aspect[1],
                    height=proj_list[0].base_aspect[0], highlightthickness=0)
    else:
        cv2.namedWindow(CV_DISP_NAME, cv2.WINDOW_NORMAL)
        print('Ready. Press any key to start.')
        cv2.waitKey(0)
        cv2.setWindowProperty(CV_DISP_NAME,
                cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print('--------------------------------')
    print('Start measurement. Don\'t touch.')
    print('--------------------------------')



    p = util.Propeller()

    for n, proj in enumerate(proj_list):
        print('----- %d/%d -----' % (n + 1, len(proj_list)))
        
        config_sub = {}
        config_sub['x_num_pixel'] = int(proj.aspect[1])
        config_sub['y_num_pixel'] = int(proj.aspect[0])
        config_sub['x_starting'] = int(proj.upperleft[1])
        config_sub['y_starting'] = int(proj.upperleft[0])
        
        
        # grey color
        grey = proj.gen_canvas()
        if BGR == True:
            grey[:] = GREY_VALUE
        else:
            grey[:, :, 1] = GREY_VALUE
        disp_img = proj.add_base(grey)
        if save_pattern:
            filename = path + 'proj_gray_proj%d_grey.png' % (n + 1)
            cv2.imwrite(filename, disp_img)

        if TK == True:
            tk_imshow(tkcanvas, disp_img)
        else:
            cv_imshow(disp_img)
        
        # adjust exposure
        print('Adjusting exposure...', end='')
        p.start()
        filename = path + 'auto_proj%d.jpg' % (n + 1)
        iso, shutter = theta.auto_adjust_exposure(filename, EV=EV)
        p.end()
        print('ISO:', iso, ', Shutter:', shutter)
        
        
        print('Caputuring...', end='')
        p.start()
        
        if PN == False:
            '''
            ポジネガモードを使わない場合，
            01判定の基準用にグレー画像を撮影。
            '''
            URI = theta.take()
            URI_list.append(URI)
            filename = 'gray_proj%d_grey.jpg' % (n + 1)
            filename_list.append(filename)
        
        # x-axis
        ret = graycodepattern(proj.aspect, axis='x', BGR=BGR)
        imgs, nbits, offset = ret
        if PN == True:
            ret = graycodepattern(proj.aspect, axis='x', BGR=BGR, reverse=True)
            imgs_nega, _, _ = ret
            
        for i in range(len(imgs)):
            
            # display & capture (posi-pattern)
            disp_img = proj.add_base(imgs[i])
            if save_pattern:
                filename = path + 'proj_gray_proj%d_x%d_posi.png' % (n + 1, i)
                cv2.imwrite(filename, disp_img)
            if TK == True:
                tk_imshow(tkcanvas, disp_img)
            else:
                cv_imshow(disp_img)

            
            URI = theta.take()
            URI_list.append(URI)
            filename = 'gray_proj%d_x%d_posi.jpg' % (n + 1, i)
            filename_list.append(filename)
            
            # display & capture (nega-pattern)
            if PN == True:
                disp_img = proj.add_base(imgs_nega[i])
                if save_pattern:
                    filename = path + 'proj_gray_proj%d_x%d_nega.png' % (n + 1, i)
                    cv2.imwrite(filename, disp_img)
                if TK == True:
                    tk_imshow(tkcanvas, disp_img)
                else:
                    cv_imshow(disp_img)

                
                URI = theta.take()
                URI_list.append(URI)
                filename = 'gray_proj%d_x%d_nega.jpg' % (n + 1, i)
                filename_list.append(filename)
                    

        config_sub['xgraycode_BGR'] = BGR
        config_sub['xgraycode_PN'] = PN
        config_sub['xgraycode_num_image'] = len(imgs)
        config_sub['xgraycode_num_bits'] = nbits
        config_sub['xgraycode_offset'] = offset

        # y-axis
        ret = graycodepattern(proj.aspect, axis='y', BGR=BGR)
        imgs, nbits, offset = ret
        if PN == True:
            ret = graycodepattern(proj.aspect, axis='y', BGR=BGR, reverse=True)
            imgs_nega, _, _ = ret
            
        for i in range(len(imgs)):
            # display & capture (posi-pattern)
            disp_img = proj.add_base(imgs[i])
            if save_pattern:
                filename = path + 'proj_gray_proj%d_y%d_posi.png' % (n + 1, i)
                cv2.imwrite(filename, disp_img)
            if TK == True:
                tk_imshow(tkcanvas, disp_img)
            else:
                cv_imshow(disp_img)

            
            URI = theta.take()
            URI_list.append(URI)
            filename = 'gray_proj%d_y%d_posi.jpg' % (n + 1, i)
            filename_list.append(filename)
            
            # display & capture (nega-pattern)
            if PN == True:
                disp_img = proj.add_base(imgs_nega[i])
                if save_pattern:
                    filename = path + 'proj_gray_proj%d_y%d_nega.png' % (n + 1, i)
                    cv2.imwrite(filename, disp_img)
                if TK == True:
                    tk_imshow(tkcanvas, disp_img)
                else:
                    cv_imshow(disp_img)

                
                URI = theta.take()
                URI_list.append(URI)
                filename = 'gray_proj%d_y%d_nega.jpg' % (n + 1, i)
                filename_list.append(filename)

        config_sub['ygraycode_BGR'] = BGR
        config_sub['ygraycode_PN'] = PN
        config_sub['ygraycode_num_image'] = len(imgs)
        config_sub['ygraycode_num_bits'] = nbits
        config_sub['ygraycode_offset'] = offset
        
        config['parameters']['projector_%d' % (n + 1)] = config_sub
        p.end()
    
    if TK == True:
        tkroot.destroy()
    else:
        cv2.destroyWindow(CV_DISP_NAME)
    
    # save config
    filename = path + 'graycode_config.json'
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
    
    
    # save images
    print('---------------')
    print('save images')
    pg = util.ProgressBar2(len(filename_list))    
    for i in range(len(filename_list)):
        theta.save(URI_list[i], path + filename_list[i])
        pg.bar()



def imread(filename):
    img = cv2.imread(filename)
    if img is None:
        raise Exception('Cannot read %s' % filename)
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
    
    print('------------------------')
    print('Analyse Graycode Pattern')
    print('------------------------')
    
    # load configuration
    filename = path + 'graycode_config.json'
    with open(filename, 'r') as f:
        config = json.load(f)
    
    img_HW = config['camera']['height'], config['camera']['width']
    
    N = config['num_projectors']
    proj_whole_HW = config['projector_whole_HW']
    proj_x_stack, proj_y_stack = [], []
    azimuth_stack, polar_stack = [], []
    overlap_x, overlap_y = [], []
    overlap_weight = []

    p = util.Propeller()
    
    for n in range(N):
        print('----- %d/%d -----' % (n + 1, N))

        proj_id = n + 1
        config_sub = config['parameters']['projector_%d' % proj_id]
        proj_HW = config_sub['y_num_pixel'], config_sub['x_num_pixel']
        x_starting = config_sub['x_starting']
        y_starting = config_sub['y_starting']

        
        scr = screen_list[n]
        i1, i2, j1, j2 = scr.get_evaluation_area_index()
        
        print('Decoding Gray-code pattern...', end=''); p.start()

        # reference image
        if not(config_sub['xgraycode_PN'] and config_sub['ygraycode_PN']):
            filename = path + 'gray_proj%d_grey.jpg' % proj_id
            img_ref = imread(filename)
            img_ref = shift_horizontal(img_ref, scr.horizontal_shift)
            img_ref = add_equirectangular_margin(
                            img_ref, margin[1], margin[0])
            img_ref = img_ref[i1:i2, j1:j2, :]

        # ----- x-axis -----
        BGR = config_sub['xgraycode_BGR']
        PN = config_sub['xgraycode_PN']
        num_imgs = config_sub['xgraycode_num_image']
        nbits = config_sub['xgraycode_num_bits']
        offset = config_sub['xgraycode_offset']
        imgs_code = np.empty([i2 - i1, j2 - j1, nbits], dtype=np.bool)
        for i in range(num_imgs):
            # load image
            filename = path + 'gray_proj%d_x%d_posi.jpg' % (proj_id, i)
            img = imread(filename)
            img = shift_horizontal(img, scr.horizontal_shift)
            img = add_equirectangular_margin(img, margin[1], margin[0])
            img = img[i1:i2, j1:j2, :]
            
            if PN == True:
                filename = path + 'gray_proj%d_x%d_nega.jpg' % (proj_id, i)
                img_nega = imread(filename)
                img_nega = shift_horizontal(img_nega, scr.horizontal_shift)
                img_nega = add_equirectangular_margin(
                                img_nega, margin[1], margin[0])
                img_nega = img_nega[i1:i2, j1:j2, :]

            # judge 0 or 1
            if BGR:
                if PN == False:
                    code = (img > img_ref)
                else:
                    code = (img > img_nega)
                for j in range(3):
                    if (3 * i + j) >= nbits: break
                    imgs_code[:, :, 3 * i + j] = code[:, :, j]
            else:
                if PN == False:
                    code = (img[:, :, 1] > img_ref[:, :, 1]) # green layer
                else:
                    code = (img[:, :, 1] > img_nega[:, :, 1])
                imgs_code[:, :, i] = code
        # decode
        imgs_bin = np.empty_like(imgs_code, dtype=np.bool)
        imgs_bin[:, :, 0] = imgs_code[:, :, 0]
        for i in range(1, nbits):
            imgs_bin[:, :, i] = np.logical_xor(
                                 imgs_bin[:, :, i - 1], imgs_code[:, :, i])
        weight = 2 ** np.arange(nbits) [::-1].reshape(1, 1, -1)
        proj_x = np.sum(imgs_bin * weight, axis=-1).astype(np.float32)
        proj_x += x_starting - offset


        # ----- y-axis -----
        BGR = config_sub['ygraycode_BGR']
        PN = config_sub['ygraycode_PN']
        num_imgs = config_sub['ygraycode_num_image']
        nbits = config_sub['ygraycode_num_bits']
        offset = config_sub['ygraycode_offset']
        imgs_code = np.empty([i2 - i1, j2 - j1, nbits], dtype=np.bool)
        for i in range(num_imgs):
            # load image
            filename = path + 'gray_proj%d_y%d_posi.jpg' % (proj_id, i)
            img = imread(filename)
            img = shift_horizontal(img, scr.horizontal_shift)
            img = add_equirectangular_margin(img, margin[1], margin[0])
            img = img[i1:i2, j1:j2, :]
            
            if PN == True:
                filename = path + 'gray_proj%d_y%d_nega.jpg' % (proj_id, i)
                img_nega = imread(filename)
                img_nega = shift_horizontal(img_nega, scr.horizontal_shift)
                img_nega = add_equirectangular_margin(img_nega, margin[1], margin[0])
                img_nega = img_nega[i1:i2, j1:j2, :]

            # judge 0 or 1
            if BGR:
                if PN == False:
                    code = (img > img_ref)
                else:
                    code = (img > img_nega)
                for j in range(3):
                    if (3 * i + j) >= nbits: break
                    imgs_code[:, :, 3 * i + j] = code[:, :, j]
            else:
                if PN == False:
                    code = (img[:, :, 1] > img_ref[:, :, 1]) # green layer
                else:
                    code = (img[:, :, 1] > img_nega[:, :, 1])
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
        
        # smoothing
        proj_x = cv2.blur(proj_x, (KSIZE_SMOOTHING_X, KSIZE_SMOOTHING_Y))
        proj_y = cv2.blur(proj_y, (KSIZE_SMOOTHING_X, KSIZE_SMOOTHING_Y))

        plt.subplot(211)
        plt.imshow(proj_x, cmap=plt.cm.jet)
        plt.subplot(212)
        plt.imshow(proj_y, cmap=plt.cm.jet)
        plt.savefig(path + 'plt_decode_%d.pdf' % proj_id)
        plt.close()
        
        
        # pixel direction
        polar, azimuth = scr.get_direction_meshgrid()

        
        # 
        index_masker = scr.get_masked_index()
        proj_x_sample = proj_x[index_masker]
        proj_y_sample = proj_y[index_masker]
        polar_sample = polar[index_masker]
        azimuth_sample = azimuth[index_masker]
        points_sample = np.c_[proj_x_sample, proj_y_sample]

        proj_x = proj_x[np.where((x_starting <= proj_x)
                        & (proj_x <= (x_starting + proj_HW[1])))]
        proj_y = proj_y[np.where((y_starting <= proj_y)
                        & (proj_y <= (y_starting + proj_HW[0])))]

        
        p.end()

        # interpolate points candidate
        print('Refining projector pixel...', end=''); p.start()
        x1 = int(np.ceil(proj_x_sample.min()))
        x2 = int(proj_x_sample.max())
        y1 = int(np.ceil(proj_y_sample.min()))
        y2 = int(proj_y_sample.max())
        proj_y_interp_cand, proj_x_interp_cand = np.mgrid[y1:y2, x1:x2]
        points_interp_cand = np.c_[
                proj_x_interp_cand.reshape(-1),
                proj_y_interp_cand.reshape(-1)
                ]
        # determine interpolate points
        n_poly = 100 # ポリゴン数（超大まかな目安）
        k = int(2 * (np.sqrt(len(points_sample)) / (n_poly / 4)) ** 2 * np.pi)
        hull = concavehull.concavehull(points_sample, k)
        inside = concavehull.check_inside(points_interp_cand, hull)
        i_inside_hull = np.where(inside)[0]
        points_interp = points_interp_cand[i_inside_hull, :]
        
        p.end()
        
        # interpolation
        print('Estimating pixel direction...', end=''); p.start()
        f = LinearNDInterpolator(points_sample, polar_sample)
        polar_interp = f(points_interp)
        
        f = LinearNDInterpolator(points_sample, azimuth_sample)
        azimuth_interp = f(points_interp)

        i_inside_area = np.where(
                (scr.area_polar[0] <= polar_interp) &
                (polar_interp <= scr.area_polar[1]) &
                (scr.area_azimuth[0] <= azimuth_interp) &
                (azimuth_interp <= scr.area_azimuth[1])
                )[0]

        proj_x_stack.append(points_interp[i_inside_area, 0])
        proj_y_stack.append(points_interp[i_inside_area, 1])
        azimuth_stack.append(azimuth_interp[i_inside_area])
        polar_stack.append(polar_interp[i_inside_area])


        # overlap weighting
        if scr.overlap_angle >= 0:
            left_side, right_side = scr.area_azimuth
            left_ovlp_end = left_side + scr.overlap_angle
            right_ovlp_end = right_side - scr.overlap_angle
            
            i_ovlp_left = np.where(
                    (left_side <= azimuth_stack[n]) & 
                    (azimuth_stack[n] < left_ovlp_end)
                    )[0]
            overlap_x.append(proj_x_stack[n][i_ovlp_left])
            overlap_y.append(proj_y_stack[n][i_ovlp_left])
            azim_ovlp = azimuth_stack[n][i_ovlp_left]
            weight = (azim_ovlp - left_side) / (left_ovlp_end - left_side)
            overlap_weight.append(weight)
            
            i_ovlp_right = np.where(
                    (right_ovlp_end < azimuth_stack[n]) & 
                    (azimuth_stack[n] <= right_side)
                    )[0]
            overlap_x.append(proj_x_stack[n][i_ovlp_right])
            overlap_y.append(proj_y_stack[n][i_ovlp_right])
            azim_ovlp = azimuth_stack[n][i_ovlp_right]
            weight = (azim_ovlp - right_side) / (right_ovlp_end - right_side)
            overlap_weight.append(weight)
        else:
            overlap_x.append([])
            overlap_y.append([])
            overlap_weight.append([])


        # cancel horizontal shift
        azimuth_stack[n] -= scr.horizontal_shift_deg   
        
        p.end()

    proj_x_stack = np.hstack(proj_x_stack).astype(np.int)
    proj_y_stack = np.hstack(proj_y_stack).astype(np.int)
    azimuth_stack = np.hstack(azimuth_stack)
    polar_stack = np.hstack(polar_stack)

    overlap_x = np.hstack(overlap_x).astype(np.int)
    overlap_y = np.hstack(overlap_y).astype(np.int)
    overlap_weight = np.hstack(overlap_weight)

    overlap_tone_input = scr.tone_input
    overlap_tone_output = scr.tone_output

    mapper = (
            proj_x_stack, proj_y_stack, polar_stack, azimuth_stack,
            overlap_x, overlap_y, overlap_weight,
            overlap_tone_input, overlap_tone_output, np.array(proj_whole_HW)
            )
    return mapper



