import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def best_fit_transform(A, B):
    """
    Tìm phép biến đổi (R,t) tối ưu để căn chỉnh A sang B.
    A, B: N x 2 điểm tương ứng.
    """
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Đảm bảo R không có phản xạ
    if np.linalg.det(R) < 0:
        Vt[1,:] *= -1
        R = Vt.T @ U.T

    t = centroid_B.T - R @ centroid_A.T
    return R, t

def icp(A, B, max_iterations=20, tolerance=1e-5):
    """
    ICP: căn chỉnh A sang B
    """
    src = np.copy(A)
    prev_error = 0

    for i in range(max_iterations):
        # 1. Tìm điểm gần nhất trong B cho mỗi điểm trong A
        tree = KDTree(B)
        distances, indices = tree.query(src)
        closest_points = B[indices]

        # 2. Tính phép biến đổi tốt nhất
        R, t = best_fit_transform(src, closest_points)

        # 3. Áp dụng phép biến đổi
        src = (R @ src.T).T + t

        # 4. Kiểm tra hội tụ
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return src, R, t

# --------- MÔ PHỎNG ---------
# Tạo đám mây điểm gốc (B)
theta = np.linspace(0, 2*np.pi, 50)
B = np.vstack((np.cos(theta), np.sin(theta))).T

# Tạo bản sao A bị xoay và dịch
angle = np.radians(30)
R_true = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
t_true = np.array([0.5, 0.2])
A = (R_true @ B.T).T + t_true

# Chạy ICP để căn chỉnh A về B
A_aligned, R_est, t_est = icp(A, B)

# Vẽ trước và sau khi căn chỉnh
plt.figure(figsize=(6,6))
plt.scatter(B[:,0], B[:,1], c='blue', label='Target B')
plt.scatter(A[:,0], A[:,1], c='red', alpha=0.5, label='Source A (trước ICP)')
plt.scatter(A_aligned[:,0], A_aligned[:,1], c='green', alpha=0.7, label='A sau ICP')
plt.legend()
plt.title("Mô phỏng ICP 2D")
plt.axis('equal')
plt.grid(True)
plt.show()
