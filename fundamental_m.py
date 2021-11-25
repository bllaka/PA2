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
            F = Evec[:, j].reshape((3, 3))
            for p in range(row):
                Q1 = np.array(pts_ah[p, :]).reshape((3, 1))
                Q2 = np.array(pts_bh[p, :]).reshape((3, 1))
                err = np.dot(np.dot(Q1.T, F), Q2)
                err = np.abs(err)
                if err < threshold:
                    numinlier += 1
                    a_inlier.append(pts_ah[p, :])
                    b_inlier.append(pts_bh[p, :])
            if numinlier > most_inliers:
                a_inlier_final = a_inlier
                b_inlier_final = b_inlier
                num_inlier = numinlier
                E_final = F
    return np.array(a_inlier_final), np.array(b_inlier_final), num_inlier, E_final


if __name__ == '__main__':
    # import image
    imga = image('dataset\mytwoview\IMG_0973.JPG')
    imgb = image('dataset\mytwoview\IMG_0974.JPG')

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(imga, cmap='Greys_r')
    ax[0].tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    ax[0].set_title('image A')
    ax[1].imshow(imgb, cmap='Greys_r')
    ax[1].tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    ax[1].set_title('image B')
    plt.tight_layout()
    plt.savefig('result/imageAB.jpg')


    # feature extraction
    kpa, da, kpb, db = ORB(imga, imgb, num_f=2000)
    # matching
    matches, pts_a, pts_b = match(da, db, kpa, kpb)
    imgmatch = cv2.drawMatchesKnn(imga, kpa, imgb, kpb, matches, None, flags=0)
    cv2.imwrite('result/mymatches.jpg', imgmatch)
    img_corr = plotCorr(imga, imgb, pts_a, pts_b)
    cv2.imwrite('result/mycorr.jpg', img_corr)

    # RANSAC with five point
    a_inlier, b_inlier, final_num_inlier, F = ransac(most_inliers=250, itr=5000,
                                                     threshold=0.05, pts_a=pts_a, pts_b=pts_b)