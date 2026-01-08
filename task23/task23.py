# task23.py
import numpy as np
import cv2

def _normalize_pts_for_F(pts):
    pts = np.asarray(pts)
    centroid = pts.mean(axis=0)
    pts_c = pts - centroid
    dists = np.sqrt((pts_c**2).sum(axis=1))
    mean_dist = dists.mean()
    scale = np.sqrt(2) / (mean_dist + 1e-12)
    T = np.eye(3)
    T[0,0] = T[1,1] = scale
    T[0,2] = -scale * centroid[0]
    T[1,2] = -scale * centroid[1]
    pts_h = np.vstack([pts.T, np.ones((1, pts.shape[0]))])
    pts_n = T @ pts_h
    return T, pts_n

def find_fundamental_matrix(shape, pts1, pts2):
    """
    8-point algorithm (normalized)
    pts1, pts2: (N,2)
    returns F (3x3)
    """
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    N = pts1.shape[0]
    if N < 8:
        raise ValueError("Need at least 8 correspondences")

    T1, p1n = _normalize_pts_for_F(pts1)  # 3xN
    T2, p2n = _normalize_pts_for_F(pts2)  # 3xN

    u = p1n[0, :] / p1n[2, :]
    v = p1n[1, :] / p1n[2, :]
    up = p2n[0, :] / p2n[2, :]
    vp = p2n[1, :] / p2n[2, :]

    A = np.zeros((N, 9))
    for i in range(N):
        A[i] = [up[i]*u[i], up[i]*v[i], up[i],
                vp[i]*u[i], vp[i]*v[i], vp[i],
                u[i],       v[i],       1.0]

    # Solve Af = 0
    _, _, Vt = np.linalg.svd(A)
    f = Vt[-1, :]
    F_hat = f.reshape(3,3)

    # Enforce rank-2
    Uf, Sf, Vtf = np.linalg.svd(F_hat)
    Sf[2] = 0.0
    F_rank2 = Uf @ np.diag(Sf) @ Vtf

    # Denormalize
    F = T2.T @ F_rank2 @ T1

    # Normalize for printing 
    if abs(F[1,1]) > 1e-12:
        F = F / F[1,1]

    return F

def compute_epipoles(F):
    """
    Return (e1, e2) homogeneous epipoles.
    e1: right nullspace of F -> F e1 = 0  (epipole in image 1)
    e2: right nullspace of F^T -> F^T e2 = 0 (epipole in image 2)
    """
    # e1: nullspace of F
    U, S, Vt = np.linalg.svd(F)
    e1 = Vt[-1, :]
    # e2: nullspace of F^T
    U2, S2, Vt2 = np.linalg.svd(F.T)
    e2 = Vt2[-1, :]

    # Normalize to make last coord = 1 if possible
    if abs(e1[-1]) > 1e-12:
        e1 = e1 / e1[-1]
    if abs(e2[-1]) > 1e-12:
        e2 = e2 / e2[-1]

    return e1, e2

def find_triangulation(K1, K2, F, pts1, pts2):
    """
    Triangulate points using E derived from F and K's.
    Returns Nx4 homogeneous point cloud (each row is [X Y Z w]).
    """
    pts1 = np.asarray(pts1).astype(float)
    pts2 = np.asarray(pts2).astype(float)

    # Essential matrix
    E = K2.T @ F @ K1

    # Enforce proper singular values (two equal, one zero)
    U, S, Vt = np.linalg.svd(E)
    s = (S[0] + S[1]) / 2.0
    E = U @ np.diag([s, s, 0.0]) @ Vt

    # Decompose
    R1, R2, t = cv2.decomposeEssentialMat(E)  # t is (3,)

    # P1 = K1 [I|0]
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3,1))])

    candidates = []
    for R in (R1, R2):
        for tt in (t, -t):
            P2 = K2 @ np.hstack([R, tt.reshape(3,1)])
            candidates.append((P1, P2, R, tt))

    def _triangulate(P1, P2, pts1, pts2):
        # cv2.triangulatePoints expects 2xN arrays
        pts1_t = pts1.T
        pts2_t = pts2.T
        Xh = cv2.triangulatePoints(P1, P2, pts1_t, pts2_t)  # 4xN
        Xh = Xh / (Xh[3:4, :] + 1e-12)
        Xh = Xh.T  # Nx4
        # compute depth in camera1 (Z after applying extrinsics)
        # For camera1: extrinsics [I|0], so camera coords equal world coords
        depth1 = Xh[:, 2]  # Z
        # For camera2, compute [R|t] from P2 = K2 [R|t] => invert K2
        Rt = np.linalg.inv(K2) @ P2
        R_cam = Rt[:, :3]
        t_cam = Rt[:, 3].reshape(3, 1)
        X_world = Xh[:, :3].T  # 3xN
        X_cam2 = R_cam @ X_world + t_cam
        depth2 = X_cam2[2, :]
        in_front = np.logical_and(depth1 > 0, depth2 > 0)
        return Xh, np.sum(in_front)

    best_X = None
    best_count = -1
    for P1c, P2c, R, tt in candidates:
        Xh, count = _triangulate(P1c, P2c, pts1, pts2)
        if count > best_count:
            best_count = count
            best_X = Xh

    return best_X  # Nx4
