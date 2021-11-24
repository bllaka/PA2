import numpy as np
from utils import *
import matplotlib.pyplot as plt
import open3d

# triangulate
# Ax = 0
# X = (x, y, z, 1) ** 3D point we need. So Ax = b
def triangulation(inpts_a, inpts_b, Pa, Pb):
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))

    # 3D points array
    X = np.zeros((3, len(inpts_a)))

    for i in range(len(inpts_a)):
        # Build A matrix:
        #     [ ax * Pa[2,0] - Pa[0,0], ax * Pa[2,1] - Pa[0,1], ax * Pa[2,2] - Pa[0,2] ],
        #     [ ay * Pa[2,0] - Pa[1,0], ay * Pa[2,1] - Pa[1,1], ay * Pa[2,2] - Pa[1,2] ],
        #     [ bx * Pb[2,0] - Pb[0,0], bx * Pb[2,1] - Pb[0,1], bx * Pb[2,2] - Pb[0,2] ],
        #     [ by * Pb[2,0] - Pb[1,0], by * Pb[2,1] - Pb[1,1], by * Pb[2,2] - Pb[1,2] ]

        # concise way to create matrix A: A matrix is minus product between matrix P column 0 and 2
        #                        [ Pa[0, 0], Pa[0, 1], Pa[0, 2] ]
        #    [ -1, 0, x ]   dot  [ Pa[1, 0], Pa[1, 1], Pa[1, 2] ]
        #    [ 0, -1, y ]        [ Pa[2, 0], Pa[2, 1], Pa[2, 2] ]

        temp1 = -np.eye(2, 3)
        temp2 = temp1
        temp1[:, 2] = inpts_a[i, :]
        temp2[:, 2] = inpts_b[i, :]
        A[0:2, :] = temp1.dot(Pa[0:3, 0:3])
        A[2:4, :] = temp2.dot(Pb[0:3, 0:3])

        # Build b vector:
        #     [ -(ax * Pa[2,3] - Pb[0,3]) ],
        #     [ -(u1.y * P1[2,3] - P1[1,3]) ],
        #     [ -(u2.x * P2[2,3] - P2[0,3]) ],
        #     [ -(u2.y * P2[2,3] - P2[1,3]) ]

        b[0:2, :] = temp1.dot(Pa[0:3, 3:4])
        b[2:4, :] = temp2.dot(Pb[0:3, 3:4])
        b *= -1

        # Solve for X vector
        cv2.solve(A, b, X[:, i:i + 1], cv2.DECOMP_SVD)
    return np.hstack((X.T, np.ones((len(inpts_a), 1))))

if __name__ == '__main__':
    # load best fundamental matrix from 5points calibration with RANSAC
    F = np.load('saved_data/F.npy')

    # inliers point in both images 241 points from 296 possible points
    # poss_a = np.load('pts_a.npy')
    # poss_b = np.load('pts_b.npy')
    inpts_a = np.load('saved_data/ptsa.npy')
    inpts_b = np.load('saved_data/ptsb.npy')
    inpts_ah = np.hstack((inpts_a, np.ones((len(inpts_a), 1))))
    inpts_bh = np.hstack((inpts_b, np.ones((len(inpts_a), 1))))

    # given K1
    K = np.array([[ 1506.070,0, 1021.043], [0, 1512.965, 698.031], [0, 0, 1 ]])

    # essential matrix E = K1.T.*F.*K2
    E = np.dot(np.dot(K.T, F), K)

    # P' = K[R|T]
    # R = UWVt
    # so we find U, Vt first: svd(E) = Udiag(1,1,0)V.T
    U, S, Vt = np.linalg.svd(E)

    # two type of W
    W = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
    Wt = W.T

    # case 1 and 2
    R1 = np.dot(np.dot(U, W), Vt)
    # case 3 and 4
    R2 = np.dot(np.dot(U, Wt), Vt)
    # last col of U = u3 == translation [t]X
    t = U[:, 2].reshape(3,1)

    # case 1 P' = [UWVt | +u3]
    Pb1 = np.dot(K, np.hstack((R1, t)))
    # case 2 P' = [UWVt | -u3]
    Pb2 = np.dot(K, np.hstack((R1, -t)))
    # case 3 P' = [UWtVt | +u3]
    Pb3 = np.dot(K, np.hstack((R2, t)))
    # case 4 P' = [UWtVt | -u3]
    Pb4 = np.dot(K, np.hstack((R2, -t)))

    # P of img A  P = [I | 0]
    Pa = np.eye(3,4)

    X1 = triangulation(inpts_a, inpts_b, Pa, Pb1)
    X2 = triangulation(inpts_a, inpts_b, Pa, Pb2)
    X3 = triangulation(inpts_a, inpts_b, Pa, Pb3)
    X4 = triangulation(inpts_a, inpts_b, Pa, Pb4)

    # X = [x, y, x, 1]T, P = [M|p4], PX = w(x, y, 1)T, sign(det(M))w / T||m3||
    # depth of points
    # dept of point confess to camera A
    PXa = np.dot(Pa, X2.T)
    w = PXa/inpts_ah.T
    M = Pa[:, :-1]
    detm = np.linalg.det(M)
    sign = np.sign(detm)
    m3 = M[-1, :]
    nm3 = np.linalg.norm(m3)
    d = w*sign / nm3*t
    maxd = np.max(d)
    print('max depth of point in front of camera A:',maxd)
    mind = np.min(d)
    print('min depth of point in front of camera A:',mind)
    # dept of point confess to camera B
    PXb = np.dot(Pb2, X2.T)
    w = PXb/inpts_bh.T
    M = Pb1[:, :-1]
    detm = np.linalg.det(M)
    sign = np.sign(detm)
    m3 = M[-1, :]
    nm3 = np.linalg.norm(m3)
    d = w*sign / nm3*t
    maxd = np.max(d)
    print('max depth of point in front of camera B:',maxd)
    mind = np.min(d)
    print('min depth of point in front of camera A:',mind)
    if maxd >= 0 and mind < 0:
        print('not case 1')
    if len(np.sign(d[1, :]) >= 0) == len(inpts_b):
        # case 4 depth of y point from camera B are always positive
        print('case 4')
        X = X4
        Pb = Pb4
    elif len(np.sign(d[1, :]) < 0) == len(inpts_b):
        # case 3 depth of y points from camera B are always negative
        print('case 3')
        X = X3
        Pb = Pb3
    else:
        print('case 2')
        X = X2
        Pb = Pb2

    X = X[:, :-1]
    # summary: case 4 [UWtVt | -u3]
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(X)
    open3d.io.write_point_cloud("./saved_data.ply", pcd)