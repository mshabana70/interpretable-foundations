import numpy as np
import matplotlib.pyplot as plt

# using the patches module to graph shapes
import matplotlib.patches as patches


# creating a square
x1y1 = [0, 0]
x1y2 = [0, 5]
x2y1 = [7, 5]
x2y2 = [7, 0]

rectangle_matrix = [x1y1, x1y2, x2y1, x2y2]

# we'll set up the plot env first
fig, ax = plt.subplots()

# now we define the shape as a patch
base_rectangle = patches.Polygon([x1y1, x1y2, x2y1, x2y2], color='salmon', ec='red', lw=2)
ax.add_patch(base_rectangle)

# scale the rectangle by 2
scale_matrix = np.array(rectangle_matrix) * 2
print(f"Orig Matrix: {rectangle_matrix}\nScaled Matrix: {scale_matrix}")
scaled_rectangle = patches.Polygon(scale_matrix, color='lightgreen', ec='green', lw=2)
ax.add_patch(scaled_rectangle)

# sheer the rectangle by shifting the x1y2 and x2y1 coordinates
sheered_matrix = np.array(rectangle_matrix) + np.array([[0, 0], [3, 0], [3, 0], [0, 0]])
print(f"Orig Matrix: {rectangle_matrix}\nSheered Matrix: {sheered_matrix}")
sheered_rectangle = patches.Polygon(sheered_matrix, color='skyblue', ec='blue', lw=2)
ax.add_patch(sheered_rectangle)

# flip the rectange over the x-axis
flipped_x_matrix = np.array(rectangle_matrix) + np.array([[0, 0], [0, -10], [0, -10], [0, 0]])
print(f"Orig Matrix: {rectangle_matrix}\nFlipped Matrix: {flipped_x_matrix}")
flipped_rectangle = patches.Polygon(flipped_x_matrix, color='salmon', ec='red', lw=2)
ax.add_patch(flipped_rectangle)

ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.set_aspect('equal') 

plt.grid(True, linestyle=':')
plt.savefig("base_rectangle.png")



