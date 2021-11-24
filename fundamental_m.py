import cv2
import numpy as np

from utils import*
import matplotlib.pyplot as plt
from calibrated_fivepoint import *

def ransac(most_inliers, itr, threshold, pts_a, pts_b):
    pts_ah = np.hstack((pts_a, np.ones((pts_a.shape[0], 1))))
    pts_bh = np.hstack((pts_b, np.ones((pts_b.shape[0], 1))))
    row, col = pts_ah.shape
    for i in range(itr):
        Evec = fivepoint(pts_ah[np.random.randint(0, row, size=5), :],
                         pts_bh[np.random.randint(0, row, size=5), :])
        for j in range(0, Evec.shape[1]):
            a_inlier = []
            b_inlier = []
            numinlier = 0
            E = Evec[:, j].reshape((3, 3))
            for p in range(row):
                Q1 = np.array(pts_ah[p, :]).reshape((3, 1))
                Q2 = np.array(pts_bh[p, :]).reshape((3, 1))
                err = np.dot(np.dot(Q1.T, E), Q2)
                err = np.abs(err)
                if err < threshold:
                    numinlier += 1
                    a_inlier.append(pts_ah[p, :])
                    b_inlier.append(pts_bh[p, :])
            if numinlier > most_inliers:
                a_inlier_final = a_inlier
                b_inlier_final = b_inlier
                num_inlier = numinlier
                E_final = E
    if a_inlier_final>0:
        return np.array(a_inlier_final), np.array(b_inlier_final), num_inlier, E_final
    else:
        return print('run RANSAC again')

if __name__ == '__main__':
    # import image
    imga = image('dataset/twoview/sfm01.jpg')
    imgb = image('dataset/twoview/sfm02.jpg')
    # feature extraction
    kpa, da, kpb, db = ORB(imga, imgb, num_f=2000)
    # matching
    matches, pts_a, pts_b = match(da, db, kpa, kpb)
    # img_corr = plotCorr(imga, imgb, pts_a, pts_b)

    # RANSAC with five point
    a_inlier, b_inlier, final_num_inlier, E = ransac(most_inliers=235, itr=10000, threshold=0.05, pts_a=pts_a, pts_b=pts_b)
