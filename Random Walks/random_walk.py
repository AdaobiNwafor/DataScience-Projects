from random import choice


class RandomWalk:

    # A class to generate random walks
    def __init__(self, num_points=50_000):

        # initialise the attributes of a walk
        self.num_points = num_points

        # all walks start at (0, 0)
        self.x_val = [0]
        self.y_val = [0]

    def fill_walk(self):
        # calculate all the points in a walk
        # keep taking steps until the walk reaches the wanted length(5000)
        while len(self.x_val) < self.num_points:

            # decide which direction to go and how far to go in that particular direction
            # right or left
            x_dir = choice([1, -1])
            x_dist = choice([0, 1, 2, 3, 4])
            x_step = x_dir * x_dist

            # up or down
            y_dir = choice([1, -1])
            y_dist = choice([0, 1, 2, 3, 4])
            y_step = y_dir * y_dist

            # rejecting moves that do not go anywhere
            if x_step == 0 and y_step == 0:
                continue

            # calculate the new position, adding them to the last added values, which is why it is -1
            x = self.x_val[-1] + x_step
            y = self.y_val[-1] + y_step

            self.x_val.append(x)
            self.y_val.append(y)
