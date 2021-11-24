import numpy as np
from calibrated_fivepoint_helper import *
from scipy.linalg import lstsq, eig

# This code was implemented from matlab to python from this work (given function calibrated_fivepoint.m):
# ARTICLE{stewenius-engels-nister-isprsj-2006,
# AUTHOR = {H. Stew\'enius and C. Engels and D. Nist\'er},
# TITLE = {Recent Developments on Direct Relative Orientation},
# JOURNAL = {ISPRS Journal of Photogrammetry and Remote Sensing},
# URL = {http://dx.doi.org/10.1016/j.isprsjprs.2006.03.005},
# VOLUME = {60},
# ISSUE = {4},
# PAGES = {284--294},
# MONTH = JUN,
# CODE = {http://vis.uky.edu/~stewe/FIVEPOINT},
# PDF = {http://www.vis.uky.edu/~stewe/publications/stewenius_engels_nister_5pt_isprs.pdf},
# YEAR = 2006

def fivepoint(a, b):
      ''' a and b is coordinate point from matching features in size [5 x 3]
      Need to convert 2 points coordinates to homogeneous coordinates first'''

      Q1 = (a[:, 0] * b[:, 0]).reshape((5, 1))
      Q2 = (a[:, 1] * b[:, 0]).reshape((5, 1))
      Q3 = (a[:, 2] * b[:, 0]).reshape((5, 1))
      Q4 = (a[:, 0] * b[:, 1]).reshape((5, 1))
      Q5 = (a[:, 1] * b[:, 1]).reshape((5, 1))
      Q6 = (a[:, 2] * b[:, 1]).reshape((5, 1))
      Q7 = (a[:, 0] * b[:, 2]).reshape((5, 1))
      Q8 = (a[:, 1] * b[:, 2]).reshape((5, 1))
      Q9 = (a[:, 2] * b[:, 2]).reshape((5, 1))
      Q = np.hstack((Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9))

      U, S, V = np.linalg.svd(Q, full_matrices=True)
      V = V.T
      EE = V[:, 5:9]
      # EE = EE.T

      A = fivepoint_helper(EE)
      A1 = A[:, :10]
      A2 = A[:, 10:]
      AA = lstsq(A1, A2)
      AAA = AA[0]
      M = -AAA[[0, 1, 2, 4, 5, 7], :]
      M0 = np.zeros((4, M.shape[1]))
      M0[0, 0] = 1
      M0[1, 1] = 1
      M0[2, 3] = 1
      M0[3, 6] = 1
      M = np.vstack((M, M0))

      D, V = eig(M)
      # D = np.real_if_close(np.diag(D))
      SOLS =   V[6:9,:]/(np.ones((3,1))*V[9,:])
      SOLS = np.vstack((SOLS, np.ones((1, 10))))
      Evec = np.matmul(EE, SOLS)
      Evecss = np.sqrt(np.sum(Evec**2, axis=0).reshape((1, 10)))
      Evecm = np.matmul(np.ones((9, 1)), Evecss)
      Evec = Evec / Evecm

      # Evec110 = Evec[1, :].reshape((1,10))
      Evec_imag = np.imag(Evec)
      whre = np.where(~Evec_imag.any(axis=0))[0]

      Evec_final = Evec[:, whre]
      Evec_final = np.real_if_close(Evec_final)
      return Evec_final

# if __name__ == '__main__':
#       a = np.random.randint(1, 1800, [5, 2])
#       a = np.hstack((a, np.ones((5,1))))
#       b = np.random.randint(1, 1800, [5, 2])
#       b = np.hstack((b, np.ones((5, 1))))
#       Evec = fivepoint(a, b)