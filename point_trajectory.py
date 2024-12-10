import matplotlib.pyplot as plt
import numpy as np
from function import isHI_WDRO 

sample_size = 100
X_initial = np.random.normal(0, 1, (sample_size, 3))

Halpern_method = isHI_WDRO(3)
Halpern_method.update(config={'eps': 0.1})
Halpern_method.fit(X_initial)
all_iterations = Halpern_method.sample_points()

fig = plt.figure(figsize=(8,8), dpi=300)
ax = fig.add_subplot(111, projection='3d')

# draw the trajectory
for i in range(sample_size):
    trajectory = np.array([iteration[i] for iteration in all_iterations])  
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], alpha=0.3, color='black')  

# draw the initial and final points
initial_points = all_iterations[0]
final_points = all_iterations[-1]
ax.scatter(initial_points[:, 0], initial_points[:, 1], initial_points[:, 2], c='dodgerblue', label='Initial Points', s=10)
ax.scatter(final_points[:, 0], final_points[:, 1], final_points[:, 2], c='darkorange', label='Final Points', s=10)


ax.legend()
ax.set_title("Visualization of Sample Point Trajectories in isHI")
plt.savefig("point_trajectory.png")
plt.show()