
# IMPORT THE RELEVANT LIBRARIES (PART 1)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# GENERATE RANDOM INPUT DATA TO TRAIN ON  (PART 2)
observations = 1000

xs = np.random.uniform(low=-10, high=10, size=(observations, 1))
zs = np.random.uniform(low=-10, high=10, size=(observations, 1))
inputs = np.column_stack((xs, zs))


# CREATE THE TARGETS WE WILL AIM AT  (PART 2)
# TARGET FUNCTION = F(X,Z) = 2*X - 3*Z + 5 + NOISE (PART 2)
# 2 and -3 are the weights and 5 is the BIAS (PART 2)
# REAL DATA ALWAYS CONTAINS NOISE, IT IS NEVER PERFECT (PART 2)

noise = np.random.uniform(-1, 1, (observations, 1))
targets = 2*xs - 3*zs + 5 + noise

print(targets.shape)


# PLOT THE TRAINING DATA (PART 3)
# the point is to see that there is a string trend that our model should learn to reproduce

targets = targets.reshape(observations,)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs, zs, targets)
ax.set_xlabel('xs')
ax.set_ylabel('zs')
ax.set_zlabel('targets')
ax.view_init(azim=100)
# plt.show()
targets = targets.reshape(observations, 1)

# INITIALIZE VARIABLES(PART 3)

init_range = 0.1
weights = np.random.uniform(-init_range, init_range, size=(2, 1))
biases = np.random.uniform(-init_range, init_range, size=1)

print(weights)
print(biases)

# SET A LEARNING RATE(PART 3)

learning_rate = 0.02
