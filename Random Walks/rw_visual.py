import matplotlib.pyplot as plt
from random_walk import RandomWalk
plt.style.available


# make a random walk
rw = RandomWalk()
rw.fill_walk()

# plotting the points in the walk
plt.style.use('classic')
fig, ax = plt.subplots(figsize=(14, 12))               # edited to set the fig size to screen res

# styling the walk
point_num = range(rw.num_points)        # added code
ax.scatter(rw.x_val, rw.y_val, c=point_num,
           cmap=plt.cm.PuBu, edgecolors='none', s=1)    # edited to include c, cmap and edgecolours

# emphasizing the start and end points of the random walk
ax.scatter(0, 0, c='green', edgecolors='none', s=20)
ax.scatter(rw.x_val[-1], rw.y_val[-1], c='red', edgecolors='none', s=20)

# remove the axis
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.show()
