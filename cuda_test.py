import numpy as np
import ctypes, cv2
import matplotlib.pyplot as plt 
from ctypes import *


def get_cuda_gray():
    dll = ctypes.CDLL('./cuda_lib.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.cuda_gray
    func.argtypes = [POINTER(c_ubyte), POINTER(c_ubyte), POINTER(c_ubyte), POINTER(c_ubyte), c_size_t]
    return func

__cuda_gray = get_cuda_gray()


def cuda_gray(a, b, c, d, size):
    a_p = a.ctypes.data_as(POINTER(c_ubyte))
    b_p = b.ctypes.data_as(POINTER(c_ubyte))
    c_p = c.ctypes.data_as(POINTER(c_ubyte))
    d_p = d.ctypes.data_as(POINTER(c_ubyte))
    __cuda_gray(a_p, b_p, c_p, d_p, size)




def get_cuda_filter():
    dll = ctypes.CDLL('./cuda_lib.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.cuda_filter
    func.argtypes = [POINTER(c_ubyte), POINTER(c_float), POINTER(c_ubyte), c_size_t, c_size_t, c_size_t]
    return func

__cuda_filter = get_cuda_filter()


def cuda_filter(gray, filter_, result, rows, columns, filterWidth):
    gray_p = gray.ctypes.data_as(POINTER(c_ubyte))
    filter_p = filter_.ctypes.data_as(POINTER(c_float))
    result_p = result.ctypes.data_as(POINTER(c_ubyte))
    __cuda_filter(gray_p, filter_p, result_p, rows, columns, filterWidth)




# testing, sum of two arrays of ones and output head part of resulting array
if __name__ == '__main__':

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

    
    gray=cv2.imread('river.jpeg', 0)
    rows, columns = gray.shape[:2]
    gray = gray.reshape((rows * columns , 1)).copy()
    gray = gray.astype('uint8')
    result = np.zeros(gray.shape).astype('uint8')
    filterWidth = 3
    filter_ = (np.ones((3,3))/9.0).astype('float32')
    
    cuda_filter(gray, filter_, result, rows, columns, filterWidth )
    print 'Done'