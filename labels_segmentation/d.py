import numpy as np
import matplotlib.pyplot as plt

# 1. Tạo đám mây điểm 2D giả lập
num_points = 150
x = np.random.uniform(0, 1, num_points)
y = np.random.uniform(0, 1, num_points)
points = np.vstack((x, y)).T

# 2. Voxel downsample 2D
voxel_size = 0.07
grid_x = np.floor(points[:,0] / voxel_size).astype(int)
grid_y = np.floor(points[:,1] / voxel_size).astype(int)
grid_hash = grid_x + grid_y * 10000
unique_indices = np.unique(grid_hash, return_index=True)[1]
points_down = points[unique_indices]

# 3. Sắp xếp điểm theo hình vuông (lưới)
# Tìm số ô theo trục x và y
n_x = grid_x.max() + 1
n_y = grid_y.max() + 1

# Tạo lưới hình vuông dựa trên grid_x và grid_y
sorted_indices = np.lexsort((grid_x[unique_indices], grid_y[unique_indices]))
points_square = points_down[sorted_indices]

# 4. Vẽ đám mây điểm dạng lưới hình vuông
plt.figure(figsize=(6,6))
plt.scatter(points[:,0], points[:,1], s=20, c='lightgray', label='Điểm gốc')
plt.scatter(points_square[:,0], points_square[:,1], s=30, c='blue', label='Sau khi giảm mẫu')
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.legend()
plt.show()
