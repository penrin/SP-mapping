import numpy as np
import matplotlib.pyplot as plt
import cv2


fname = 'path/to/mapping_table.npz'



filez = np.load(fname)
proj_HW = filez['proj_HW']
x = filez['x']
y = filez['y']
zenith = filez['polar']
azimuth = filez['azimuth']

azi = np.full(proj_HW, np.nan)
azi[y, x] = azimuth
azi_mid = (azi[:, :-1] + azi[:, 1:]) / 2
azi1 = np.c_[azi[:, :1], azi_mid]
azi2 = np.c_[azi_mid, azi[:, -1:]]

zen = np.full(proj_HW, np.nan)
zen[y, x] = zenith
zen_mid = (zen[:-1, :] + zen[1:, :]) / 2
zen1 = np.r_[zen[:1, :], zen_mid]
zen2 = np.r_[zen_mid, zen[-1:, :]]

font = cv2.FONT_HERSHEY_SIMPLEX



image = np.zeros([proj_HW[0], proj_HW[1], 3])
image[y, x] = 50



print('Used pixels:')
print('   %d/%d' % (len(x), proj_HW[0] * proj_HW[1]), end='')
print(' ({:.1%})'.format(len(x) / (proj_HW[0] * proj_HW[1])))



zen_grid_list = np.linspace(0, 180, 180 // 5 + 1, endpoint=True)
azi_grid_list = np.linspace(-180, 360, (360 + 180) // 5 + 1, endpoint=True)

def intersect2d(idx1, idx2):
    set1 = set()
    for i in range(len(idx1[0])):
        set1.add((idx1[0][i], idx1[1][i]))
    set2 = set()
    for i in range(len(idx2[0])):
        set2.add((idx2[0][i], idx2[1][i]))

    r, c = [], []

    for rc in (set1 & set2):
        r.append(rc[0])
        c.append(rc[1])
    
    return np.array(r), np.array(c)


print('----- Zenith -----')

with np.errstate(invalid='ignore'):
    idx_text_align_azi = np.where((azi1 <= 0) & (0 < azi2))

for zen_grid in zen_grid_list:
    with np.errstate(invalid='ignore'):
        idx = np.where((zen1 <= zen_grid) & (zen_grid < zen2))
        if len(idx[0]) == 0:
            continue
    
    # grid
    if zen_grid == 90:
        image[idx] = np.array([0, 165, 255])
    else:
        image[idx] = 255

    # text
    idx_text = intersect2d(idx, idx_text_align_azi)
    r, c = idx_text[0][0], idx_text[1][0] 
    cv2.putText(image, '%d' % len(idx[0]), (c, r), font, 0.5, (0, 165, 255), 1)

    print('%3d deg: %5d px' % (zen_grid, len(idx[0])), end='')
    if zen_grid == 90:
        print(' *')
    else:
        print()


print('----- Azimuth -----')

with np.errstate(invalid='ignore'):
    idx_text_align_zen = np.where((zen1 <= 90) & (90 < zen2))

for azi_grid in azi_grid_list:
    with np.errstate(invalid='ignore'):
        idx = np.where((azi1 <= azi_grid) & (azi_grid < azi2))
        if len(idx[0]) == 0:
            continue
    
    # grid
    if azi_grid == 0:
        image[idx] = np.array([255, 255, 0])
    else:
        image[idx] = 255


    # text
    idx_text = intersect2d(idx, idx_text_align_zen)
    r, c = idx_text[0][0], idx_text[1][0]
    cv2.putText(image, '%d' % len(idx[0]), (c, r), font, 0.5, (255, 255, 0), 1)
    print('%3d deg: %5d px' % (azi_grid, len(idx[0])))




cv2.imwrite('statistic.png', image)





