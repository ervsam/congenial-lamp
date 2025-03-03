# MAP GENERATION

import numpy as np
import random
from scipy.ndimage import label

# set random seed
random.seed(0)
np.random.seed(0)

def generate_map(size_x, size_y, obstacle_density):
    total_elements = size_x * size_y
    num_ones = int(total_elements * obstacle_density)

    # Initialize the array with all zeros
    binary_array = np.zeros((size_x, size_y), dtype=int)

    # Randomly place 1's in the array while keeping track of the placed positions
    ones_positions = set()
    while len(ones_positions) < num_ones:
        x = random.randint(0, size_x - 1)
        y = random.randint(0, size_y - 1)
        if (x, y) not in ones_positions:
            binary_array[x, y] = 1
            ones_positions.add((x, y))

            # Check connectivity of zeros
            labeled_array, num_features = label(binary_array == 0)
            if num_features > 1:
                # If zeros are not connected, revert the placement
                binary_array[x, y] = 0
                ones_positions.remove((x, y))

    # pad with ones
    binary_array = np.pad(binary_array, 1, 'constant', constant_values=1)
    return binary_array
