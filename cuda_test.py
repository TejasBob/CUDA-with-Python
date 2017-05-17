import numpy as np
import ctypes, cv2
import matplotlib.pyplot as plt 
from ctypes import *

# extract  function pointer in the shared object cuda_sum.so
def get_cuda_gray():
    dll = ctypes.CDLL('./cuda_lib.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.cuda_gray
    func.argtypes = [POINTER(c_ubyte), POINTER(c_ubyte), POINTER(c_ubyte), POINTER(c_ubyte), c_size_t]
    return func

# create  function with get_cuda_sum()
__cuda_gray = get_cuda_gray()

# convenient python wrapper for __cuda_sum
# it does all job with types convertation
# from python ones to C++ ones
def cuda_gray(a, b, c, d, size):
    a_p = a.ctypes.data_as(POINTER(c_ubyte))
    b_p = b.ctypes.data_as(POINTER(c_ubyte))
    c_p = c.ctypes.data_as(POINTER(c_ubyte))
    d_p = d.ctypes.data_as(POINTER(c_ubyte))
    __cuda_gray(a_p, b_p, c_p, d_p, size)



# testing, sum of two arrays of ones and output head part of resulting array
if __name__ == '__main__':

    # file_dir = os.path.dirname(os.path.realpath(__file__))
    # image_path = os.path.join(file_dir, 'images')

    # image_name = os.path.join(image_path, 'river.jpg')
    img = cv2.imread('river.jpeg')
    rows, columns = img.shape[:2]
    b,g,r = cv2.split(img)
    gray = np.zeros((rows*columns, 1))

    b = np.copy(b.reshape((rows * columns , 1))).astype('uint8')
    g = np.copy(g.reshape((rows * columns , 1))).astype('uint8')
    r = np.copy(r.reshape((rows * columns , 1))).astype('uint8')
    gray = np.copy(gray).astype('uint8')

    cuda_gray(b, g, r, gray, rows * columns)
    gray = gray.reshape(rows, columns,)
    cv2.imwrite('gray.jpg', gray)
    # print 'Difference with OpenCV output : ', np.sum(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) - gray)

