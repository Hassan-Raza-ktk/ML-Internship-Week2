
# Week 2 Python Fundamentals For ML "HASSAN RAZA"

# Task 2.1: Numpy arrays Operations

# --------Step 1------------Install NumPy: pip install numpy 2. Create numpy_basics.py file 3. Import numpy as np 

import numpy as np
print("Output Step 3: Numpy Installed Successfully ",np.__version__,"\n")

# --------Step 4------------Create arrays using np.array(), np.zeros(), np.ones(), np.arange()

# One Dimension array
arr1 = np.array([1, 2, 3, 4, 5])
print("Output Step 4: One Dimension Array:", arr1,"\n")

# Zeros array
zeros = np.zeros((2, 3))
print("Zeros Array:\n", zeros,"\n")
# Ones array
ones = np.ones((3, 2))
print("Ones Array:\n", ones,"\n")

# Arange
arange = np.arange(0, 10, 2)
print("Arange Array:", arange,"\n")


# --------Step 5------------Demonstrate reshaping with .reshape()

arr2 = np.arange(12)
reshaped = arr2.reshape(3, 4)
print("Output Step 5: Reshaped Array (3x4):\n", reshaped,"\n")

# --------Step 6------------Show slicing operations

sliced = reshaped[1:, 2:]  # Rows 1 to end, columns 2 to end
print("Output Step 6: Sliced Array:\n", sliced,"\n")

# --------Step 7------------Perform mathematical operations (add, multiply, dot product)

a = np.array([1, 2])
b = np.array([3, 4])

# Addition
Add = a + b
print("Output Step 7: Addition:", Add)

# Multiplication

Multiply = a * b
print("Multiplication:", Multiply)

# Dot Product
Dot = np.dot(a, b)
print("Dot Product:", Dot,"\n")

# --------Step 8------------Calculate statistics (mean, median, std, variance)

arr = np.array([1, 2, 3, 4, 5])
print("Output Step 8: Mean:", np.mean(arr))
print("Median:", np.median(arr))
print("Standard Deviation:", np.std(arr))
print("Variance:", np.var(arr),"\n")

# --------Step 9------------Use broadcasting

arr = np.array([1, 2, 3, 4, 5])
broadcasted = arr + 10
print("Output Step 9: Broadcasted Array:", broadcasted,"\n")

# --------Step 10------------Save outputs as screenshots


# 11. Commit and push to GitHub