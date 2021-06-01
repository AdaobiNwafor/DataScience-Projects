import matplotlib.pyplot as plt
from random_walk import RandomWalk
plt.style.available

# keep making new walks as long as the program is active
while True:
    # Make a random walk
    rw = RandomWalk()
    rw.fill_walk()

    plt.style.use('classic')
    fig, ax = plt.subplots()
    ax.scatter(rw.x_val, rw.y_val, s=5)
    plt.show()

    keep_running = input('Make another walk? (y/n)?: ')
    if keep_running == 'n':
        break
