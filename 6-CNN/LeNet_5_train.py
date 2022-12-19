import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import LeNet_5_inference


BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./LeNet-5_model/"
MODEL_NAME = "LeNet-5_model"


def train(mnist):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, LeNet_5_inference.IMAGE_SIZE, LeNet_5_inference.IMAGE_SIZE,
                                    LeNet_5_inference.NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, LeNet_5_inference.OUTPUT_NODE], name='y-input')

    # l2正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 前向传播
    y = LeNet_5_inference.inference(x, True,regularizer)
    # 训练轮数
    global_step = tf.Variable(0, trainable=False)

    # 滑动平均
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 学习率衰减
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,  # 过完所有训练数据，需要的轮数
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 模型持久化
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (
            BATCH_SIZE, LeNet_5_inference.IMAGE_SIZE, LeNet_5_inference.IMAGE_SIZE, LeNet_5_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
