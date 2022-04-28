
# Use of Numpy for slicing
import numpy as np

# An array for example
arr = np.array([[-1, 5, 0, 4],
[8, -0.5, 6, 0],
[2.6, 0, 8, 8],
[1, -7, 4, 4.0]])

# Slicingthe array
temp = arr[:2, ::2]
print ("Array with first 2 rows and alternate"
"columns(0 and 2):\n", temp)