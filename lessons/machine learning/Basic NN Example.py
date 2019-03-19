
# IMPORT THE RELEVANT LIBRARIES (PART 1)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# GENERATE RANDOM INPUT DATA TO TRAIN ON
observations = 1000

xs = np.random.uniform(low=-10, high=10, size=(observations, 1))
zs = np.random.uniform(low=-10, high=10, size=(observations, 1))
inputs = np.column_stack((xs, zs))

print(inputs.shape)

# CREATE THE TARGETS WE WILL AIM AT
# TARGET FUNCTION = F(X,Z) = 2*X - 3*Z + 5 + NOISE
# 2 and -3 are the weights and 5 is the BIAS
# REAL DATA ALWAYS CONTAINS NOISE, IT IS NEVER PERFECT
