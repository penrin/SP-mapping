import sys
import cv2
import numpy as np
from scipy import interpolate
import time
from math import ceil



class ProgressBar():

    def __init__(self, bar_length=40, slug='#', space='-', countdown=True):

        self.bar_length = bar_length
        self.slug = slug
        self.space = space
        self.countdown = countdown
        self.start_time = None
        self.start_parcent = 0
    
    def bar(self, percent, end=1, tail=''):
        percent = percent / end

        if self.countdown == True:
            progress = percent - self.start_parcent
            
            if self.start_time == None:
                self.start_time = time.perf_counter()
                self.start_parcent = percent
                remain = 'Remain --:--:--'
                
            elif progress == 0:
                remain = 'Remain --:--:--'
            
            elif progress != 0:
                elapsed_time = time.perf_counter() - self.start_time
                progress = percent - self.start_parcent
                remain_t = (elapsed_time / progress) * (1 - percent)
                remain_t = ceil(remain_t)
                h = remain_t // 3600
                m = remain_t % 3600 // 60
                s = remain_t % 60
                remain = 'Remain %02d:%02d:%02d' % (h, m, s) 
                
        else:
            remain = ''
        
        len_slugs = int(percent * self.bar_length)
        slugs = self.slug * len_slugs
        spaces = self.space * (self.bar_length - len_slugs)
        txt = '\r[{bar}] {percent:.1%} {remain} {tail}'.format(
                bar=(slugs + spaces), percent=percent,
                remain=remain, tail=tail)
        if percent == 1:
            txt += '\n'
            self.start_time = None
        sys.stdout.write(txt)
        sys.stdout.flush()
        


class ProgressBar2(ProgressBar):

    def __init__(self, end, bar_length=40, slug='#', space='-', countdown=True):
        
        super().__init__(bar_length=40, slug='#', space='-', countdown=True)
        self.counter = 0
        self.end = end
        self.bar()
        
    def bar(self, tail=''):
        super().bar(self.counter, end=self.end, tail='')
        self.counter += 1


class Timer():
    def __init__(self):
        self.start_time = 0
        self.stop_time = 0
    def start(self):
        self.start_time = time.time()
    def stop(self):
        self.stop_time = time.time()
        print('time: %.2f' % (self.stop_time - self.start_time))



def convert_image():

    sph_HW = img.shape[0], img.shape[1]
    proj_x = mapper['x']
    proj_y = mapper['y']
    sph_x = mapper['azimuth'] * sph_HW[1] / 360
    sph_y = mapper['polar'] * sph_HW[0] / 180

    sx1 = (np.floor(sph_x) + 1).astype(np.int32)
    sx2 = np.floor(sph_x).astype(np.int32)
    sy1 = (np.floor(sph_y) + 1).astype(np.int32)
    sy2 = np.floor(sph_y).astype(np.int32)
    sx1d = sx1 - sph_x
    sx2d = sph_x - sx2
    sy1d = sy1 - sph_y
    sy2d = sph_y - sy2
    w1 = (sx2d * sy2d).reshape(-1, 1)
    w2 = (sx2d * sy1d).reshape(-1, 1)
    w3 = (sx1d * sy2d).reshape(-1, 1)
    w4 = (sx1d * sy1d).reshape(-1, 1)
    sx1 %= sph_HW[1]
    sx2 %= sph_HW[1]
    sy1 %= sph_HW[0]
    sy2 %= sph_HW[0]
    
    b = np.zeros([sph_HW[0], sph_HW[1]], dtype=np.bool)
    b[sy1, sx1] = True
    b[sy2, sx1] = True
    b[sy1, sx2] = True
    b[sy2, sx2] = True
    ii = np.where(b)
    index_table = np.zeros([sph_HW[0], sph_HW[1]], dtype=np.int64)
    index_table[ii[0], ii[1]] = np.arange(len(ii[0]))
    i1 = index_table[sy1, sx1]
    i2 = index_table[sy2, sx1]
    i3 = index_table[sy1, sx2]
    i4 = index_table[sy2, sx2]
    
    # gamma
    data = (img[ii[0], ii[1], :] / 255) ** gamma
    
    # mapping
    data = data[i1, :] * w1 + data[i2, :] * w2\
         + data[i3, :] * w3 + data[i4, :] * w4
    
    # degamma
    data = data ** (contrast / gamma)
    data[np.where(data > 1.)] = 1.

    data_out = np.zeros([proj_HW[0], proj_HW[1], 3])
    data_out[proj_y, proj_x] = data
    
    # overlap
    if len(mapper['ovlp_weight']) > 0:
        overlap = True
        ovlp_w = mapper['ovlp_weight'].reshape(-1, 1)
        ovlp_x = mapper['ovlp_x']
        ovlp_y = mapper['ovlp_y']
        tone_input = mapper['tone_input'] / 255
        tone_output = mapper['tone_output'] / 255
        f_i2o = interpolate.interp1d(tone_input, tone_output, kind='cubic')
        f_o2i = interpolate.interp1d(tone_output, tone_input, kind='cubic')
        
        data_ovlp = data_out[ovlp_y, ovlp_x]
        data_out[ovlp_y, ovlp_x] = f_o2i(f_i2o(data_ovlp) * ovlp_w)
        
    data_out *= 255
    data_out.astype(np.uint8)

    cv2.imwrite(outfilename, data_out)

    

def convert_video():

    # ----- バイリニア補間前準備 -----
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sph_HW = H, W

    proj_x = mapper['x']
    proj_y = mapper['y']
    sph_x = mapper['azimuth'] * sph_HW[1] / 360
    sph_y = mapper['polar'] * sph_HW[0] / 180
    
    sx1 = (np.floor(sph_x) + 1).astype(np.int32)
    sx2 = np.floor(sph_x).astype(np.int32)
    sy1 = (np.floor(sph_y) + 1).astype(np.int32)
    sy2 = np.floor(sph_y).astype(np.int32)

    # 重みのインデックス
    #（重みは256ステップ。LUTで変換。）
    sx1d = sx1 - sph_x
    sx2d = sph_x - sx2
    sy1d = sy1 - sph_y
    sy2d = sph_y - sy2
    w1 = (sx2d * sy2d * 255).reshape(-1, 1, 1).astype(np.uint8)
    w2 = (sx2d * sy1d * 255).reshape(-1, 1, 1).astype(np.uint8)
    w3 = (sx1d * sy2d * 255).reshape(-1, 1, 1).astype(np.uint8)
    w4 = (sx1d * sy1d * 255).reshape(-1, 1, 1).astype(np.uint8)
    
    # 
    sx1 %= sph_HW[1]
    sx2 %= sph_HW[1]
    sy1 %= sph_HW[0]
    sy2 %= sph_HW[0]
    
    b = np.zeros([sph_HW[0], sph_HW[1]], dtype=np.bool)
    b[sy1, sx1] = True
    b[sy2, sx1] = True
    b[sy1, sx2] = True
    b[sy2, sx2] = True
    ii = np.where(b)
    index_table = np.zeros([sph_HW[0], sph_HW[1]], dtype=np.int64)
    index_table[ii[0], ii[1]] = np.arange(len(ii[0]))
    i1 = index_table[sy1, sx1]
    i2 = index_table[sy2, sx1]
    i3 = index_table[sy1, sx2]
    i4 = index_table[sy2, sx2]
    
    
    # ----- オーバーラップ前準備 -----
    overlap = False
    if len(mapper['ovlp_weight']) > 0:
        overlap = True
        ovlp_w = (mapper['ovlp_weight'] * 255).reshape(-1, 1, 1).astype(np.uint8)
        ovlp_x = mapper['ovlp_x']
        ovlp_y = mapper['ovlp_y']
        
        proj_ = np.zeros(proj_HW, dtype=np.int64)
        proj_[proj_y, proj_x] = np.arange(len(sy1))
        ovlp_i = proj_[ovlp_y, ovlp_x]
    
 
    
    # ----- 各種ルックアップテーブル作成 -----
    if LUT_quality == 'uint16':
        # gamma
        LUT_gamma = (((np.arange(256) / 255) ** gamma) * 60000).astype(np.uint16)
        # bilinear weight
        LUT_bili = np.empty([60001, 256], dtype=np.uint16)
        arr = np.arange(60001)
        for i in range(256):
            LUT_bili[:, i] = arr * (i / 255)
        # de-gamma
        LUT_degamma = (
                (np.arange(65536) / 60000) ** (contrast / gamma) * 60000
                ).astype(np.uint16)
        LUT_degamma[60000:] = 60000
        # overlap weight
        if overlap:
            tone_input = mapper['tone_input'] / 255 * 60000
            tone_output = mapper['tone_output'] / 255 * 60000
            f_i2o = interpolate.interp1d(tone_input, tone_output, kind='cubic')
            f_o2i = interpolate.interp1d(tone_output, tone_input, kind='cubic')
            LUT_ovlp = np.zeros([60001, 256], dtype=np.uint16)
            arr = np.arange(60001)
            for i in range(1, 256):
                w = i / 255
                LUT_ovlp[:, i] = f_o2i(f_i2o(arr) * w)
        # LUT export
        LUT_16to8 = (np.arange(60001) / 60000 * 255).astype(np.uint8)
        
    elif LUT_quality == 'uint8':
        # gamma
        LUT_gamma = (((np.arange(256) / 255) ** gamma) * 255).astype(np.uint8)
        # bilinear weight
        LUT_bili = np.empty([256, 256], dtype=np.uint8)
        arr = np.arange(256)
        for i in range(256):
            LUT_bili[:, i] = arr * (i / 255)
        # de-gamma
        LUT_degamma = (
                (np.arange(256) / 255) ** (contrast / gamma) * 255
                ).astype(np.uint16)
        # overlap weight
        if overlap:
            tone_input = mapper['tone_input']
            tone_output = mapper['tone_output']
            f_i2o = interpolate.interp1d(tone_input, tone_output, kind='cubic')
            f_o2i = interpolate.interp1d(tone_output, tone_input, kind='cubic')
            LUT_ovlp = np.zeros([256, 256], dtype=np.uint8)
            arr = np.arange(256)
            for i in range(1, 256):
                w = i / 255
                LUT_ovlp[:, i] = f_o2i(f_i2o(arr) * w)
    else:
        raise Exception('LUT_quality support uint8 or uint16')
    
    

    print('Converting')
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    nframes = 120
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(outfilename, fourcc, fps, (proj_HW[1], proj_HW[0]))
    
    nbuff = 10

    if LUT_quality == 'uint16':
        buff_i = np.empty([len(ii[0]), 3, nbuff], dtype=np.uint16)
    elif LUT_quality == 'uint8':
        buff_i = np.empty([len(ii[0]), 3, nbuff], dtype=np.uint8)
    buff_o = np.zeros([proj_HW[0], proj_HW[1], 3, nbuff], dtype=np.uint8)
    
 
    # ----- 変換 -----
    print('nframe: %d' % nframes)

    N = int(np.ceil(nframes / nbuff))
    pg = ProgressBar2(N)
    
    tt = False
    if tt: t = Timer()

    for n in range(N):
        
        if n == (N - 1):
            L = nframes % nbuff
            if L == 0:
                L = nbuff
        else:
            L = nbuff
        
        # read
        if tt: 
            print()
            print('read')
            t.start()
        for i in range(L):
            ret, frame = cap.read()
            buff_i[:, :, i] = frame[ii[0], ii[1], :]
        if tt: t.stop()
        
        # gamma
        if tt:
            print('gamma')
            t.start()
        buff_i = LUT_gamma[buff_i]
        if tt: t.stop()
        
        # mapping
        if tt:
            print('mapping')
            t.start()
        buff_o1 = LUT_bili[buff_i[i1], w1] + LUT_bili[buff_i[i2], w2]\
                + LUT_bili[buff_i[i3], w3] + LUT_bili[buff_i[i4], w4]
        if tt: t.stop()

        # 1 / gamma
        if tt:
            print('de-gamma')
            t.start()
        buff_o1 = LUT_degamma[buff_o1]
        if tt: t.stop()
        
        # overlap
        if overlap:
            if tt:
                print('overlap')
                t.start()
            buff_o1[ovlp_i, :, :] = LUT_ovlp[buff_o1[ovlp_i, :, :], ovlp_w]
            if tt: t.stop()
        
        # write
        if tt:
            print('write')
            t.start()
        if LUT_quality == 'uint16':
            buff_o[proj_y, proj_x, :, :] = LUT_16to8[buff_o1]
        elif LUT_quality == 'uint8':
            buff_o[proj_y, proj_x, :, :] = buff_o1
        for i in range(L):
            writer.write(buff_o[:, :, :, i])
        if tt: t.stop()
    
        pg.bar()
        
    writer.release()


    
    
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

    
    path = '../../workfolder_1'
    if path[-1] != '/':
        path += '/'
    
    mapper = np.load(path + 'mapping_table.npz')    

    infilename = '/Users/penrin/Desktop/sample_2.JPG'
    outfilename = path + 'output.jpg'
    infilename = '/Users/penrin/Desktop/190712_YANMAR/Garmin/V1780536.MP4'
    outfilename = path + 'output_123.mp4'
    
    
    gamma = 2.2
    contrast = 1.0
    LUT_quality = 'uint16' # 'uint8'
    proj_img = cv2.imread(path + 'projector_1.png')
    proj_HW = proj_img.shape[0], proj_img.shape[1]
    

    img = cv2.imread(infilename)
    if img is not None:
        convert_image()
    else:
        cap = cv2.VideoCapture(infilename)
        convert_video()
        cap.release()
    
