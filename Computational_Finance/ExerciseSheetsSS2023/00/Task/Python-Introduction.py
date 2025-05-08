# Python Introduction - First steps
# This is a script that will help you to get to know the basic elements of Python
# Please replace all ??? within this file with your solutions

# Please make sure you followed the instructions on how to install Python and please make sure the following packages are correctly installed: Numpy, Scipy, Matplotlib


# 1. Importing packages
# using the import-statement to get access to functions from different packages
# different ways to import packages
# import the whole package
import math
# import the whole package with a new name
import numpy as np
import matplotlib.pyplot as plt

# 2. Real variables and basic mathematical operations
# set x to 5 and y to 3 and fill in the correct operations
x ???
y ???
print('The sum of x and y is: ' + str(???))
print('The difference of x and y is: ' + str(???))
print('The product of x and y is: ' + str(???))
print('The quotient of x and y is: ' + str(???))

# Hint: for the following you can use the power-function from the math package
print('x to the power of 1.5 is: ' + str(???))

# 3. Array initialization
# In this section you will learn how to initialize Arrays and basic calculations on it.
# Initialize X as an Array with place for ten floating point numbers using the empty-function from the numpy package
X ???
# now save in X the vector [9,8,7,6,5,4,3,2,1,0]
X ???
# initialize Y as the following Array [0 1 2 3 4 5 6 7 8 9]
Y ???
# transpose X
???
# calculate the scalarproduct of X and Y
print('The scalarproduct of X and Y is: ' + str(???))
# initialize Z as array of ones with size (1,5)
Z ???
# multiply Z by 3
print(???)
# initialize V as array of zeros with size (5,1)
V ???
print(V)
# initialize R as the range from 8 to 24 (including 24) with a stepsize of 2
R ???
# print the fourth entry of R
print('The fourth entry of R is: ' + str(???))
# initialize L as the linespace from -2.5 to 2.5 with 50 samples
L ???
# print the 20th entry of L
print('The 20th entry of L is: ' + str(???))

# 4. Matrices
# define the matrix consisting of 2*X, X+Y and Y^3 (use the numpy power-function)
M ???
print(M)
# create a matrix containing ones with size (10,3)
O ???
# multiply M with O, use either the dot-function or the matmul-function from the numpy package
print('The matrixproduct of M and O is: ' + str(???))
# multiply M with Y
print('The product of M and Y is: ' + str(???))
# create a 5x5 matrix filled with N(0,1)-distributed values
# Hint: Check out the random-functions of numpy
G ???
print(G)

# accessing array-elements
# complete the following statements
print('The 5th element in the 5th row of G is: ' + str(???))
print('The first row of G is: ' + str(???))
print('The third column of G is: ' + str(???))

# 5. Branching
# True-False Statements
# Give a True statement for each operator using x and y: <, >=, ==, !=
print(???)
print(???)
print(???)
print(???)

# Write an if-else statement which compares z to x and y and prints one of the following messages
z = 4
???
    print('z is equal to x')
???
    print('z is equal to y')
???
    print('z is not equal to x nor y')

# 6. For-loop
# Use a for-loop to calculate the sum of the first 50 even numbers (starting with 0)
sum = 0
???
print('The sum of the first 50 even numbers is: ' + str(sum))


# 7. Custom functions
# define a function polynom which calculates the result of p^2+3p+2
???

print(polynom(5))


# implement the function h(x,y) = (x/y, exp(x))
???

print(h(1, 1))

# 8. Importing data
# load DAX data using the genfromtxt-function from numpy
# we're only interested in the value and not the date, therefore only import the fourth column and skip the header
dax = ???

# 9. Flip the data
# since the timeseries is antichronological you need to flip it using the flip-function from numpy.
dax = ???

# 10. Plotting
# plot the DAX data and label the axis
???
