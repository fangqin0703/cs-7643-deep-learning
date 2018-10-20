import math
import numpy as np

def sigmoid(x):
	return 1.0/(1 + math.exp(-x))

def f1(w1, w2):
	x = math.exp(w1) + math.exp(2.0*w2)
	print(math.exp(x))
	return math.exp(x) + math.sin(x)
	
def f2(w1, w2):
	return w1*w2 + sigmoid(w1)
	
if __name__ == "__main__":
	w1 = 1
	w2 = 2
	
	t1 = f1(w1, w2)
	print(t1)
	t2 = f2(w1, w2)
	print(t2)
	
	del_w1 = 0.01
	del_w2 = 0.01
	J = np.zeros((2, 2))
	J[0,0] = (f1(w1 + del_w1, w2) - f1(w1, w2)) / del_w1
	J[0,1] = (f1(w1, w2 + del_w2) - f1(w1, w2)) / del_w2
	J[1,0] = (f2(w1 + del_w1, w2) - f2(w1, w2)) / del_w1
	J[1,1] = (f2(w1, w2 + del_w2) - f2(w1, w2)) / del_w2
	print(J)