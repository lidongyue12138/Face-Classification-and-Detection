'''
This is the code for Model Construction: SVM Model
'''
# necessary imports
import tensorflow as tf
import numpy as np

class VanillaCNN:
    def __init__(self):

        self.sess = tf.Session()
        self.build_model()
        self.sess.run(self.init)

    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 96, 96, 3])
        self.y_true = tf.placeholder(tf.float32, shape=[None, 2])

        conv1 = self.conv_layer(self.x, shape=[5, 5, 3, 32])
        conv1_pool = self.max_pool_2x2(conv1)

        conv2 = self.conv_layer(conv1_pool, shape=[5, 5, 32, 64])
        conv2_pool = self.max_pool_2x2(conv2)

        conv2_flat = tf.reshape(conv2_pool, shape=[-1, 24*24*64])
        full_1 = tf.nn.relu(self.full_layer(conv2_flat, 1024))

        # keep_prob = tf.placeholder(tf.float32)
        # full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

        y_conv = self.full_layer(full_1, 2)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=y_conv, labels=self.y_true
        ))

        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        self.correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y_true, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.init = tf.global_variables_initializer()

    def train_model(self, X, y):
        y = self.one_hot(y)
        feed_dict = {self.x: X, self.y_true: y}
        
        self.sess.run(self.train_step, feed_dict=feed_dict)
        [loss, accuracy] = self.sess.run(
            [self.cross_entropy, self.accuracy],
            feed_dict = feed_dict
        )
        return loss, accuracy

    def test_acc(self, X, y):
        y = self.one_hot(y)
        feed_dict = {self.x: X, self.y_true: y}
        
        [loss, accuracy] = self.sess.run(
            [self.cross_entropy, self.accuracy],
            feed_dict = feed_dict
        )
        print("Testing accuracy: %f" %accuracy)

    '''
    Helper Functions: to build model more conveniently
    '''
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def conv_layer(self, input, shape):
        W = self.weight_variable(shape)
        b = self.bias_variable([shape[3]])
        return tf.nn.relu(self.conv2d(input, W) + b)

    def full_layer(self, input, size):
        in_size = int(input.get_shape()[1])
        W = self.weight_variable([in_size, size])
        b = self.bias_variable([size])
        return tf.matmul(input, W) + b

    def one_hot(self, vec, vals=2):
        n = len(vec)
        out = np.zeros((n, vals))
        out[range(n), vec] = 1
        return out

if __name__ == "__main__":
    cnn = VanillaCNN()