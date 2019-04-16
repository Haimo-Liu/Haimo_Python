import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np


x1 = tf.constant([3, 5, 12])
y1 = tf.Variable(x1 + 5)


#y2 = tf.Variable(tf.reduce_mean(x2))



for i in range(5):

    x2 = tf.constant(np.random.randint(0, 1000, 100))
    y2 = tf.Variable(tf.reduce_mean(x2))

    initializer = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(initializer)
        print(session.run(y2))

