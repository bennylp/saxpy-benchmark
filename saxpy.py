N = 2 ** 26
XVAL = 2.5
YVAL = 1.5
AVAL = 3.5

def verify(y):
    import numpy as np
    err = np.sum(np.abs(np.array(y) - (AVAL * XVAL + YVAL)))
    print("Error: {}".format(err))
    return err == 0.0
