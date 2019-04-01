import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

img = plt.imread("E:\\file\\11111.png")

f = np.array([[-1, 0, 1], [-2, 0, 2]])


inp = np.array([[1, 2], [3, 4]], np.float32)
H1, W1 = inp.shape[:2]
K = np.array([[-1, -2], [2, 1]], np.float32)
H2, W2 = K.shape[:2]
c_full = signal.convolve2d(inp, K, mode='full')
kr, kc = 0, 0
c_same = c_full[H2 - kr - 1:H1 + H2 - kr - 1, W2 - kc - 1:W1 + W2 - kc - 1]
print(c_same)
