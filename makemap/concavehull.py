import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import sys
import time


def concavehull(points, k, visual=False):
    kk = max(k, 3)
    dataset = np.unique(points, axis=0).astype(np.float)
    if len(dataset) < 3:
        raise Exception('A minimum of 3 dissimilar points is required')
    if len(dataset) == 3:
        return dateset

    if visual:
        plt.close()
        plt.plot(dataset[:, 0], dataset[:, 1], '.', alpha=0.5)
    
    
    kk = min(kk, len(dataset) - 1)
    
    index_first_point = np.argmin(dataset[:, 1])
    first_point = np.copy(dataset[index_first_point, :])
    hull = [first_point]

    current_index = index_first_point
    current_point = np.copy(first_point)
    dataset[index_first_point, :] = np.nan
    prev_angle = 0
    step = 2
    while ((current_point.tolist() != first_point.tolist()) or (step == 2))\
            and (np.sum(~np.isnan(dataset)) > 0):
        
        if step == 5:
            dataset[index_first_point, :] = first_point
        
        # candidate
        k_nearest_index = nearest_points(dataset, current_point, kk)
        k_nearest_points = dataset[k_nearest_index, :]
        c_points_index = sort_by_angle(k_nearest_points, current_point, prev_angle)
        
        # select the first candidate that does not intersects
        # any of the polygon edges.
        its = True
        i = 0
        while (its == True) and (i < len(c_points_index)):
            i += 1
            if k_nearest_points[c_points_index[i - 1]].tolist() == first_point.tolist():
                last_point = 1
            else:
                last_point = 0
            j = 2
            its = False
            p1 = current_point
            p2 = k_nearest_points[c_points_index[i - 1]]
            while (its == False) and (j < (len(hull) - last_point)):
                p3 = hull[step - 2 - j]
                p4 = hull[step - 1 - j]
                its = check_intersects((p1, p2), (p3, p4))
                j += 1
                
        if its == True:
            # since all candidates intersect at least one edge,
            # try again with a higher number of nighbours.
            if visual:
                pass
            print('try again with k + %d (intersect)' % (kk + 1))
            return concavehull(points, kk + 1, visual=visual)
        
        if visual:
            plt.title('k:%d, step:%d' % (kk, step))
            pp1, = plt.plot(current_point[0], current_point[1], '.', color='C1')
            pp2, = plt.plot(k_nearest_points[:, 0], k_nearest_points[:, 1], '.', color='k')
    
        
        current_point[:] = k_nearest_points[c_points_index[i - 1]]
        hull.append(np.copy(current_point[:]))
        prev_angle = calc_prev_angle(hull[step - 1], hull[step - 2])
        
        index_current = k_nearest_index[c_points_index[i - 1]]
        dataset[index_current, :] = np.nan
        
        if visual:
            hull_ = toArray_hull(hull)
            plt.plot(dataset[:, 0], dataset[:, 1], '.', alpha=0.5)
            pp3, = plt.plot(hull_[:-1, 0], hull_[:-1, 1], '-', color='C1')
            pp4, = plt.plot(hull_[-2:, 0], hull_[-2:, 1], ':', color='C1')
            plt.axis('equal')
            plt.pause(0.001)
            pp1.remove()
            pp2.remove()
            pp3.remove()
            pp4.remove()
            
        step += 1
    
    hull_ = toArray_hull(hull)
    inside = check_inside(dataset, hull_, nan_is=True, radius=0.01)
    n_outside = np.sum(~inside)
    all_inside = (n_outside == 0)
    
    if all_inside == False:
        if True:
            # since at least one point is out of the computed polygon,
            # try again with a higher number of nighbours.
            pp1, = plt.plot(hull_[:, 0], hull_[:, 1], '-', color='C1')
            i_outside = np.where(~inside)[0]
            pp2, = plt.plot(dataset[i_outside, 0], dataset[i_outside, 1], 'kx')
            plt.pause(0.1)
            pp1.remove()
            pp2.remove()
            
        print('try again with k + %d (outside %d)' % (kk + 1, n_outside))
        return concavehull(points, kk + 1, visual=visual)
    
    if visual:
        plt.plot(hull_[:, 0], hull_[:, 1], '-', color='C1')
        plt.fill(hull_[:, 0], hull_[:, 1], color='C1', alpha=0.3)
        plt.axis('equal')
        plt.pause(0.5)
        plt.show()
    
    return hull_

    

def nearest_points(dataset, current_point, kk):
    r = np.sum((dataset - current_point) ** 2, axis=1)
    k_nearest_index = np.argpartition(r, kk)[:kk]
    isnotnan = np.where(~np.isnan(dataset[k_nearest_index, 0]))
    return k_nearest_index[isnotnan]

def sort_by_angle(points, current_point, prev_angle):
    # sort the candidates(neighbours) in descending order of right-hand turn
    v = points - current_point
    angles = (np.arctan2(v[:, 1], -v[:, 0]) - prev_angle) % (2 * np.pi)
    return np.argsort(-angles)

def check_intersects(edge1, edge2, endpoint=False):
    p1, p2 = edge1
    p3, p4 = edge2
    t1 = (p1[0] - p2[0]) * (p3[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[0])
    t2 = (p1[0] - p2[0]) * (p4[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[0])
    t3 = (p3[0] - p4[0]) * (p1[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p1[0])
    t4 = (p3[0] - p4[0]) * (p2[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p2[0])
    if endpoint == False:
        # 端点を含まない
        ans = (t1 * t2 < 0) & (t3 * t4 < 0)
    else:
        # 端点を含む
        ans = (t1 * t2 <= 0) & (t3 * t4 <= 0)
    return ans


def calc_prev_angle(current_point, prev_point):
    v = prev_point - current_point
    return np.arctan2(v[1], -v[0])

    
def toArray_hull(hull):
    hull_ = np.empty([len(hull), 2])
    for i in range(len(hull)):
        hull_[i, :] = hull[i]
    return hull_
    

def check_inside(points, hull, nan_is=False, radius=0.0):

    p = Path(hull, closed=True)
    inside = p.contains_points(points, radius=radius)

    points_nan = np.isnan(points[:, 0]) | np.isnan(points[:, 1])
    i_nan = np.where(points_nan)[0]
    if (len(i_nan) > 0) & (nan_is == True):
        inside[i_nan] = True

    return inside

    

if __name__ == '__main__':
    
    N = 500
    np.random.seed(seed=0)
    x = np.random.randn(N)
    y = np.random.randn(N) - np.cos(x)
    
    points = np.c_[x, y]
    k = 3
    hull = concavehull(points, k, visual=True)
    



