# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

LEARNING_RATE = 0.01
NUM_INPUT = 784
NUM_HIDDEN_1 = 256
NUM_HIDDEN_2 = 64
NUM_HIDDEN_3 = 16
NUM_HIDDEN_4 = 2

EPOCHS = 40
BATCH_SIZE = 256
DISPLAY_STEP = 100


class AutoEncoder():

    def __init__(self):
        # Initialize paras
        with tf.name_scope("Weights"):
            self.W = {
                'encoder_1': tf.Variable(tf.truncated_normal([NUM_INPUT, NUM_HIDDEN_1], stddev=0.1), name='W1'),
                'encoder_2': tf.Variable(tf.truncated_normal([NUM_HIDDEN_1, NUM_HIDDEN_2], stddev=0.1), name='W2'),
                'encoder_3': tf.Variable(tf.truncated_normal([NUM_HIDDEN_2, NUM_HIDDEN_3], stddev=0.1), name='W3'),
                'encoder_4': tf.Variable(tf.truncated_normal([NUM_HIDDEN_3, NUM_HIDDEN_4], stddev=0.1), name='W4'),
                'decoder_1': tf.Variable(tf.truncated_normal([NUM_HIDDEN_4, NUM_HIDDEN_3], stddev=0.1), name='W5'),
                'decoder_2': tf.Variable(tf.truncated_normal([NUM_HIDDEN_3, NUM_HIDDEN_2], stddev=0.1), name='W6'),
                'decoder_3': tf.Variable(tf.truncated_normal([NUM_HIDDEN_2, NUM_HIDDEN_1], stddev=0.1), name='W7'),
                'decoder_4': tf.Variable(tf.truncated_normal([NUM_HIDDEN_1, NUM_INPUT], stddev=0.1), name='W8'),
            }
        with tf.name_scope('biases'):
            self.b = {
                'encoder_1': tf.Variable(tf.random_normal([NUM_HIDDEN_1]), name='b1'),
                'encoder_2': tf.Variable(tf.random_normal([NUM_HIDDEN_2]), name='b2'),
                'encoder_3': tf.Variable(tf.random_normal([NUM_HIDDEN_3]), name='b3'),
                'encoder_4': tf.Variable(tf.random_normal([NUM_HIDDEN_4]), name='b4'),
                'decoder_1': tf.Variable(tf.random_normal([NUM_HIDDEN_3]), name='b5'),
                'decoder_2': tf.Variable(tf.random_normal([NUM_HIDDEN_2]), name='b6'),
                'decoder_3': tf.Variable(tf.random_normal([NUM_HIDDEN_1]), name='b7'),
                'decoder_4': tf.Variable(tf.random_normal([NUM_INPUT]), name='b8'),
            }

    def encoder(self, inp):
        layer_1 = tf.nn.sigmoid(
            tf.add(tf.matmul(inp, self.W['encoder_1']), self.b['encoder_1']))
        layer_2 = tf.nn.sigmoid(
            tf.add(tf.matmul(layer_1, self.W['encoder_2']), self.b['encoder_2']))
        layer_3 = tf.nn.sigmoid(
            tf.add(tf.matmul(layer_2, self.W['encoder_3']), self.b['encoder_3']))
        # Linear decoder
        layer_4 = tf.add(
            tf.matmul(layer_3, self.W['encoder_4']), self.b['encoder_4'])

        return layer_4

    def decoder(self, inp):
        layer_1 = tf.nn.sigmoid(
            tf.add(tf.matmul(inp, self.W['decoder_1']), self.b['decoder_1']))
        layer_2 = tf.nn.sigmoid(
            tf.add(tf.matmul(layer_1, self.W['decoder_2']), self.b['decoder_2']))
        layer_3 = tf.nn.sigmoid(
            tf.add(tf.matmul(layer_2, self.W['decoder_3']), self.b['decoder_3']))
        layer_4 = tf.nn.sigmoid(
            tf.add(tf.matmul(layer_3, self.W['decoder_4']), self.b['decoder_4']))

        return layer_4

    def train(self):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, NUM_INPUT], name='X')
        with tf.name_scope('output'):
            y = tf.placeholder(tf.float32, [None, NUM_INPUT], name='Y')

        output_encoder = self.encoder(x)
        output_y = self.decoder(output_encoder)

        # Define loss and optimizer
        loss = tf.reduce_mean(tf.pow(y - output_y, 2))
        optimizer = tf.train.AdamOptimizer(
            learning_rate=LEARNING_RATE).minimize(loss)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(output_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Training loop
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            total_batch = int(mnist.train.num_examples / BATCH_SIZE)

            for epoch in range(1, EPOCHS + 1):
                for step in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
                    sess.run(optimizer, feed_dict={x: batch_x, y: batch_x})
                    if step % DISPLAY_STEP == 0:
                        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={
                                                     x: batch_x, y: batch_x})
                        print('Epoch:', epoch, '| Step:', step,
                              '| Test loss:', loss_val, '|Test accuracy:', acc_val)

            print('Accuracy:', 1 -
                  accuracy.eval({x: mnist.test.images, y: mnist.test.images}))
            print('Finished!')

            # Show results
            SHOW_NUM = 10
            reconstruction = sess.run(
                output_y, feed_dict={x: mnist.test.images[:SHOW_NUM]})
            figure, a = plt.subplots(2, 10, figsize=(10, 2))
            plt.title('Results')
            for i in range(SHOW_NUM):
                a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
                a[1][i].imshow(np.reshape(reconstruction[i], (28, 28)))
            plt.show()

            aa = [np.argmax(a) for a in mnist.test.labels]  # onehot -> 一般编码
            encoder_result = sess.run(output_encoder, feed_dict={
                                      x: mnist.test.images})
            plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=aa)
            plt.colorbar()
            plt.title('2-Dim Feature')
            plt.show()

if __name__ == '__main__':
    AutoEncoder().train()
