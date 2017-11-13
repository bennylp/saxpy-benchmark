import math

N = 2 ** 26
XVAL = 2.0
YVAL = 1.0
AVAL = 3.0

def saxpy_verify(y):
    err = 0.0
    for i in range(N):
        err += abs(y[i] - (AVAL * XVAL + YVAL))
    print("Error: {}".format(err))
    return err == 0.0
