import numpy as np
import utils


import numpy as np

def _normalize_points_2d(p):
    # p: (N,2) or (2,N) -> return T (3x3), pn (3xN)
    p = np.asarray(p)
    if p.ndim == 1:
        p = p[np.newaxis, :]
    if p.shape[1] == 2:
        pts = p.T  # 2xN
    else:
        pts = p.T[:2,:]
    N = pts.shape[1]
    centroid = pts.mean(axis=1, keepdims=True)
    pts_cent = pts - centroid
    dists = np.sqrt((pts_cent**2).sum(axis=0))
    mean_dist = dists.mean()
    scale = np.sqrt(2) / (mean_dist + 1e-12)
    T = np.eye(3)
    T[0,0] = T[1,1] = scale
    T[0,2] = -scale * centroid[0,0]
    T[1,2] = -scale * centroid[1,0]
    pts_h = np.vstack([pts, np.ones((1, N))])
    pts_n = T @ pts_h
    return T, pts_n

def _normalize_points_3d(X):
    # X: (N,3) -> return T (4x4), Xn (4xN)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[np.newaxis, :]
    pts = X.T[:3,:]  # 3xN
    centroid = pts.mean(axis=1, keepdims=True)
    pts_cent = pts - centroid
    dists = np.sqrt((pts_cent**2).sum(axis=0))
    mean_dist = dists.mean()
    scale = np.sqrt(3) / (mean_dist + 1e-12)
    T = np.eye(4)
    T[:3,:3] *= scale
    T[:3,3] = -scale * centroid.flatten()
    pts_h = np.vstack([pts, np.ones((1, pts.shape[1]))])
    Xn = T @ pts_h
    return T, Xn

def _build_A(Xh, ph):
    # Xh: 4xN homogeneous 3D, ph: 3xN homogeneous 2D
    N = Xh.shape[1]
    A = np.zeros((2*N, 12))
    for i in range(N):
        X_i = Xh[:,i]   # 4,
        u = ph[0,i] / ph[2,i]
        v = ph[1,i] / ph[2,i]
        A[2*i, 0:4] = X_i
        A[2*i, 8:12] = -u * X_i
        A[2*i+1, 4:8] = X_i
        A[2*i+1, 8:12] = -v * X_i
    return A

def find_projection(pts2d, pts3d):
    """
    pts2d: (N,2), pts3d: (N,3)
    returns P: 3x4
    """
    pts2d = np.asarray(pts2d)
    pts3d = np.asarray(pts3d)
    N = pts2d.shape[0]
    if pts3d.shape[0] != N:
        raise ValueError("Number of 2D and 3D points must match")

    # Homogeneous forms and normalize
    T2, p_n = _normalize_points_2d(pts2d)   # p_n: 3 x N
    T3, X_n = _normalize_points_3d(pts3d)   # X_n: 4 x N

    A = _build_A(X_n, p_n)
    # A m = 0 via SVD
    _, _, Vt = np.linalg.svd(A)
    m = Vt[-1, :]
    Mnorm = m.reshape(3,4)

    # Denormalize: want P so that p ~ P X ; with normalization,
    # p_n = T2 p, X_n = T3 X -> T2 p ~ Mnorm (T3 X) => p ~ inv(T2) Mnorm T3 X
    P = np.linalg.inv(T2) @ Mnorm @ T3

    # Normalize scale for consistency 
    if abs(P[2,3]) > 1e-12:
        P = P / P[2,3]
    else:
        P = P / np.linalg.norm(P)

    return P


def reprojection_error(P, pts3d, pts2d):
    Xh = np.hstack([pts3d, np.ones((pts3d.shape[0],1))]).T  # 4xN
    proj = (P @ Xh)
    proj = proj / proj[2,:]
    proj_xy = proj[:2,:].T
    diffs = proj_xy - pts2d
    dists = np.sqrt((diffs**2).sum(axis=1))
    return dists.mean(), dists


if __name__ == '__main__':
    pts2d = np.loadtxt("task1/pts2d.txt")
    pts3d = np.loadtxt("task1/pts3d.txt")

    # Alternately, for some of the data, we provide pts1/pts1_3D, which you
    # can check your system on via
    """
    data = np.load("task23/ztrans/data.npz")
    pts2d = data['pts1']
    pts3d = data['pts1_3D']
    """

