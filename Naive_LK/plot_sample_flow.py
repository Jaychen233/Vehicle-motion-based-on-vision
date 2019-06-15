'''
Plot theoretical optical flow
=============================

Plots the theoretical optical flow induced by a camera moving parallel to a
plane that's orthogonal to the image plane.
'''

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from IPython import embed

y_size = 10

fig = plt.figure(figsize=(6, 4), dpi=80)
ax = fig.add_subplot(111)

flow_x = lambda x, y: 0.01 * (x * y)
flow_y = lambda _, y: 0.01 * (y * y)

for x in range(-15, 15, 2):
    for y in range(y_size, 0, -2):
        ax.arrow(x, y_size - y, flow_x(x, y), -flow_y(x, y), head_length=0.3, head_width=0.2, ec='k', fc='k')
        # ax.add_line(Line2D([x, x+flow_x(x, y)], [y_size - y, y_size - y - flow_y(x, y)]))

ax.set_xlim(-15, 15)
ax.set_ylim(-2, 10)

ax.axis('off')

ax.figure.savefig('flow_example.png', bbox_inches='tight')

# embed()

plt.show()
