# distutils: language = c++
# distutils: sources = src/csa.cc src/Exception.cc src/kofn.cc src/match.cc src/Matrix.cc src/Random.cc src/String.cc src/Timer.cc
# distutils: extra_compile_args = -DNOBLAS
#
import math
import numpy as np


cdef extern from "../src/Matrix.hh":
    cppclass Matrix:
        Matrix () except +
        Matrix (int rows, int cols) except +
        Matrix (int rows, int cols, double* data) except +
        double* data ()

cdef extern from "../src/match.hh":
    double matchEdgeMaps(const Matrix& bmap1, const Matrix& bmap2,
                         double maxDist, double outlierCost,
                         Matrix& match1, Matrix& match2)



cdef _correspond_pixels(double[::1,:] img0, double[::1,:] img1, double max_dist, double outlier_cost,
                        double[::1,:] out0, double[::1,:] out1):
    cdef int rows = img0.shape[0]
    cdef int cols = img0.shape[1]
    cdef double idiag = math.sqrt(rows * rows + cols * cols)
    cdef double oc = outlier_cost * max_dist * idiag

    # Copy data to Matrix types; construct matrices, get views of their contents and copy
    # over
    # Constructing a Matrix from a double* acquired from views of im0 and img1 don't
    # work well at all...
    cdef Matrix i0 = Matrix(rows, cols)
    cdef Matrix i1 = Matrix(rows, cols)
    cdef double[::1,:] i0_view = <double[:img0.shape[0]:1,:img0.shape[1]]>i0.data()
    cdef double[::1,:] i1_view = <double[:img1.shape[0]:1,:img1.shape[1]]>i1.data()
    i0_view[:,:] = img0[:,:]
    i1_view[:,:] = img1[:,:]

    # Output matrices
    cdef Matrix m0, m1

    # Perform the match
    cdef double cost = matchEdgeMaps(i0, i1, max_dist * idiag, oc, m0, m1)

    # Get views of the output matrices and copy to our output arrays
    cdef double[::1,:] o0_view = <double[:out0.shape[0]:1,:out0.shape[1]]>m0.data()
    cdef double[::1,:] o1_view = <double[:out1.shape[0]:1,:out1.shape[1]]>m1.data()
    out0[:,:] = o0_view[:,:]
    out1[:,:] = o1_view[:,:]

    return cost, oc


def correspond_pixels(img0, img1, max_dist=0.0075, outlier_cost=100.0):
    if img0.shape != img1.shape:
        raise ValueError('img0.shape ({}) and img1.shape({}) do not match'.format(img0.shape, img1.shape))
    if max_dist <= 0.0:
        raise ValueError('max_dist must be >= 0 (it is {})'.format(max_dist))
    if outlier_cost <= 1:
        raise ValueError('outlier_cost must be > 1 (it is {})'.format(max_dist))


    i0 = img0.astype('float64').copy(order='F')
    i1 = img1.astype('float64').copy(order='F')
    o0 = np.zeros_like(i0, order='F')
    o1 = np.zeros_like(i1, order='F')
    max_dist = float(max_dist)
    outlier_cost = float(outlier_cost)
    cost, oc = _correspond_pixels(i0, i1, max_dist, outlier_cost, o0, o1)
    return o0, o1, cost, oc