import sys
import time

import saxpy
import tensorflow as tf

def main(where):
    print("Using " + where)
    print("N: {}".format(saxpy.N))
    Y = tf.Variable(tf.constant(saxpy.YVAL, shape=[saxpy.N], dtype=tf.float32), name="Y")
    A = tf.constant(saxpy.AVAL, tf.float32, name="A")
    X = tf.Variable(tf.constant(saxpy.XVAL, shape=[saxpy.N], dtype=tf.float32), name="X")
    saxpy_node = tf.assign_add(Y, A * X)

    verbose = False
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True) if verbose else None)

    init = tf.global_variables_initializer()
    sess.run(init)

    t0 = time.time()
    sess.run(saxpy_node)
    t1 = time.time()
    print("Elapsed: {} ms".format((t1 - t0) * 1000.0))

    answer = tf.constant(saxpy.AVAL * saxpy.XVAL + saxpy.YVAL, tf.float32, name="Answer")
    err_node = tf.reduce_sum(tf.abs(Y - answer))
    print("Errors: {}".format(sess.run(err_node)))


if __name__ == "__main__":
    where = sys.argv[1] if len(sys.argv) > 1 else ""
    if where == "gpu":
        with tf.device("/device:GPU:0"):
            main("/device:GPU:0")
    elif where == "cpu":
        with tf.device("/device:CPU:0"):
            main("/device:CPU:0")
    else:
        main("default device")
