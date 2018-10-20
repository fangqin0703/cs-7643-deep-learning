import math
import numpy as np

def sigmoid(x):
	return 1.0/(1 + math.exp(-x))

def my_fun(x):
	return sigmoid(math.log(5.0*(max(x[0], x[1])*x[2]/x[3] - (x[4] + x[5]))) + 1/2)
	
if __name__ == "__main__":
	x = [5, -1, 6, 12, 7, -5]
	print(my_fun(x))
	
	eps = 1e-5
	x1 = np.add(x, [eps, 0, 0, 0, 0, 0])
	print(x1)
	x1_grad = (my_fun(x1) - my_fun(x)) / eps
	print(x1_grad)
	
	x2 = np.add(x, [0, eps, 0, 0, 0, 0])
	x2_grad = (my_fun(x2) - my_fun(x)) / eps
	print(x2_grad)
	
	x3 = np.add(x, [0, 0, eps, 0, 0, 0])
	x3_grad = (my_fun(x3) - my_fun(x)) / eps
	print(x3_grad)
	
	x4 = np.add(x, [0, 0, 0, eps, 0, 0])
	x4_grad = (my_fun(x4) - my_fun(x)) / eps
	print(x4_grad)
	
	x5 = np.add(x, [0, 0, 0, 0, eps, 0])
	x5_grad = (my_fun(x5) - my_fun(x)) / eps
	print(x5_grad)
	
	x6 = np.add(x, [0, 0, 0, 0, 0, eps])
	x6_grad = (my_fun(x6) - my_fun(x)) / eps
	print(x6_grad)