import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

'''
one_hot means:
10 classes: 0-9

0 = [1,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0]

'''

#build a computational graph, x1, x2, result are all tensors...connected by the computation (edges)
#this now becomes a graph
# x1 = tf.constant(5)
# x2 = tf.constant(6)
# result = tf.multiply(x1, x2)

#run a graph in a session
# with tf.Session() as sess:
#     output = sess.run(result)
#     print(output)


# input -> weight -> hidden layer 1 (activation function) -> hidden layer 2
# repeat this process....weights to the output layer....
# compare outpu to intended output...with a cost function (cross entropy)
#  optimizer --> minize the cost (adam Optimizer...SGD, AdaGrad are some other options....)
#
# backpropagation...
#
# feed forward + backprop = epoch (lowering cost function in each iteration)



