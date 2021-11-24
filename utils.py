import numpy as np
import cv2

def image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = img.astype(np.float32) / 255
    return img

def ORB(imga, imgb, num_f):
    orb = cv2.ORB_create(nfeatures=num_f)
    kp_a, d_a = orb.detectAndCompute(imga, None)
    kp_b, d_b = orb.detectAndCompute(imgb, None)
    return kp_a, d_a, kp_b, d_b

def concatImg(imga, imgb):
    ha = imga.shape[0]
    hb = imgb.shape[0]
    wa = imga.shape[1]
    wb = imgb.shape[1]
    h = max(ha, hb)
    w = wa + wb
    newimg = np.zeros((h, w), dtype=imga.dtype)
    newimg[:ha, :wa] = imga
    newimg[:hb, wa:] = imgb
    return newimg

def match(da, db, kpa, kpb):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(da, db, k=2)
    good = []
    for m,n in matches:
        if m.distance < n.distance/1.2:
            good.append([m])
    pts_a = []
    pts_b = []
    for i in good:
        pts_a.append(kpa[i[0].queryIdx].pt)
        pts_b.append(kpb[i[0].trainIdx].pt)
    pts_a = np.array(pts_a).astype(np.int)
    pts_b = np.array(pts_b).astype(np.int)
    return good, pts_a, pts_b

def plotCorr(imga, imgb, pts_a, pts_b):
    img = concatImg(imga, imgb)
    xa = pts_a[:, 0]
    ya = pts_a[:, 1]
    xb = pts_b[:, 0]
    yb = pts_b[:, 1]

    dot_color = np.random.rand(len(xa), 3)
    dot_color *= 255
    line_color = dot_color
    xs = imga.shape[1]

    for xa, ya, xb, yb, dc, lc in zip(xa, ya, xb, yb, dot_color, line_color):
        img = cv2.circle(img, (xa, ya), 3, dc, -1)
        img = cv2.circle(img, (xb + xs, yb), 3, dc, -1)
        img = cv2.line(img, (xa, ya), (xb + xs, yb), lc, 1, cv2.LINE_AA)
    return img