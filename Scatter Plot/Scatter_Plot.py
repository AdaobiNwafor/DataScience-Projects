import numpy as np
import matplotlib.pyplot as plt

input_val = [1,2,3,4,5]
squares = [1,4,9,16,25]

# customisation of the plot
plt.style.use('seaborn')
# plt.style.use('ggplot')
plt.style.use('seaborn-muted')
plt.style.use('seaborn-pastel')

# making the figure
fig, ax = plt.subplots()
ax.plot(input_val, squares, linewidth=2)

ax.set_title('Squares Numbers', fontsize=14)
ax.set_xlabel('Values', fontsize=12)
ax.set_ylabel('Square of numbers', fontsize=12)

ax.tick_params(axis='both', labelsize=12)

plt.show()


# plotting a series of points with scatter
x_val = [1,2,3,4,5]
y_val = [1,4,9,16,25]

fig, ax = plt.subplots()
ax.scatter(x_val, y_val, s=75)

ax.set_title('Squares Numbers', fontsize=14)
ax.set_xlabel('Values', fontsize=12)
ax.set_ylabel('Square of numbers', fontsize=12)

plt.show()


# calculating data automatically
x_val2 = range(1, 1001)
y_val2 = [x**2 for x in x_val2]
fig, ax = plt.subplots()
# ax.scatter(x_val2, y_val2, s=5)

# change the colour of the points
ax.scatter(x_val2, y_val2, c='red', s=5)      # similar to code above

ax.set_title('Squares Numbers', fontsize=14)
ax.set_xlabel('Values', fontsize=10)
ax.set_ylabel('Square of numbers', fontsize=12)

# set the range for the axis
ax.set_xlim(0, 1100)
ax.set_ylim(0, 1100000)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)

plt.show()


# third scatter graph
xval3 = range(-1101, 1101)
yval3 = [x**3 for x in xval3]
fig, ax = plt.subplots()

# using a colour map with the colour set Blues
ax.scatter(xval3, yval3, c=yval3, cmap=plt.cm.Blues, s=5)

ax.set_title('Cube Numbers', fontsize=14)
ax.set_xlabel('Values', fontsize=10)
ax.set_ylabel('Cube of numbers', fontsize=12)

# set the range for the axis
ax.set_xlim(-1100, 1100)
ax.set_ylim(-1000000000, 1000000000)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)

plt.savefig('cubes_plot.png', bbox_inches='tight')
