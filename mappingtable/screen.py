import sys
import os



class Screen():
    
    def __init__(self, filename):
        
        png = cv2.imread(filename)
        if png is None:
            print(filename, 'could not read.')
            sys.exit()
            


def set_config(path2work):
    
    screen_list = []
    num = 0
    
    while True:
        num += 1
        filename = path2work + 'screen_%d.png' % num
        if os.path.isfile(filename):
            scr = Screen(filename)
            screen_list.append(scr)
        else:
            if num == 1:
                print(filename, 'is not exits.')
                print('Please put the screen cofiguration png.')
                sys.exit()
            break
    
    return screen_list
    
    

