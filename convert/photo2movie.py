import sys
import cv2
import argparse
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



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='output filename')
    parser.add_argument('-i', type=str, help='input image filename', required=True)
    parser.add_argument('--nframes', type=int, default=30, help='number of frames')
    parser.add_argument('--fps', type=float, default=30, help='fps')
    args = parser.parse_args()
    
    filename_img = args.i
    filename_mov = args.filename
    nframes = args.nframes
    fps = args.fps
    
    img = cv2.imread(filename_img)

    if img is None:
        print('image cannot be read')
        sys.exit()

    shape = img.shape[1], img.shape[0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename_mov, fourcc, fps, shape)

    pg = ProgressBar2(nframes)
    
    for i in range(nframes):
        writer.write(img)
        pg.bar()

    writer.release()



