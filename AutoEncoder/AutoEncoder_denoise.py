# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

LEARNING_RATE = 0.01
NUM_INPUT = 784
NUM_HIDDEN_1 = 256
NUM_HIDDEN_2 = 128

EPOCHS = 10
BATCH_SIZE = 256
DISPLAY_STEP = 100


class AutoEncoder():

	def __init__(self):
		# 初始化模型参数
		self.keep_prob = 0.5
		with tf.name_scope("Weights"):
			self.W = {
			'encoder_1': tf.Variable(tf.truncated_normal([NUM_INPUT, NUM_HIDDEN_1], stddev=0.1), name='W1'),
			'encoder_2': tf.Variable(tf.truncated_normal([NUM_HIDDEN_1, NUM_HIDDEN_2], stddev=0.1), name='W2'),
			'decoder_1': tf.Variable(tf.truncated_normal([NUM_HIDDEN_2, NUM_HIDDEN_1], stddev=0.1), name='W3'),
			'decoder_2': tf.Variable(tf.truncated_normal([NUM_HIDDEN_1, NUM_INPUT], stddev=0.1), name='W4'),
			}
		with tf.name_scope('biases'):
			self.b = {
				'encoder_1': tf.Variable(tf.random_normal([NUM_HIDDEN_1]), name='b1'),
				'encoder_2': tf.Variable(tf.random_normal([NUM_HIDDEN_2]), name='b2'),
				'decoder_1': tf.Variable(tf.random_normal([NUM_HIDDEN_1]), name='b3'),
				'decoder_2': tf.Variable(tf.random_normal([NUM_INPUT]), name='b4'),
			}

	def encoder(self, inp):
		layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(inp, self.W['encoder_1']), self.b['encoder_1']))
		layer_1 = tf.nn.dropout(layer_1, self.keep_prob)
		layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.W['encoder_2']), self.b['encoder_2']))
		layer_2 = tf.nn.dropout(layer_2, self.keep_prob)


		return layer_2

	def decoder(self, inp):
		layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(inp, self.W['decoder_1']), self.b['decoder_1']))
		layer_1 = tf.nn.dropout(layer_1, self.keep_prob)
		layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.W['decoder_2']), self.b['decoder_2']))

		return layer_2

	def train(self):
		with tf.name_scope('input'):
			x = tf.placeholder(tf.float32, [None, NUM_INPUT], name='X')
		with tf.name_scope('output'):
			y = tf.placeholder(tf.float32, [None, NUM_INPUT], name='Y')
		keep_prob = tf.placeholder(tf.float32)

		output_encoder = self.encoder(x)   
		output_y = self.decoder(output_encoder)

		# Define loss and optimizer
		loss = tf.reduce_mean(tf.pow(y - output_y, 2))
		optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

		# Evaluate model
		correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(output_y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		# Training loop
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			total_batch = int(mnist.train.num_examples/BATCH_SIZE)

			for epoch in range(1, EPOCHS+1):
				for step in range(total_batch):
					batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
					batch_x_noise = batch_x + 0.5*np.random.randn(BATCH_SIZE, 784)
					sess.run(optimizer, feed_dict={x: batch_x, y: batch_x, keep_prob: self.keep_prob})
					if step % DISPLAY_STEP == 0:
						loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x: batch_x, y: batch_x, keep_prob:1.})
						print('Epoch:', epoch, '| Step:', step, '| Test loss:',loss_val, '|Test accuracy:', acc_val)
			
			print('Accuracy:', 1 - accuracy.eval({x:mnist.test.images, y:mnist.test.images}))
			print('Finished!')		

			# Show results
			SHOW_NUM = 10

			# image = self.get_a_image('./1.png')
			test_x = mnist.test.images[:SHOW_NUM] + 0.5*np.random.randn(SHOW_NUM, 784)
			reconstruction = sess.run(output_y, feed_dict={x: test_x, keep_prob:1.})
			# reconstruction = sess.run(output_y, feed_dict={x: image})
			figure, a = plt.subplots(3, 10, figsize=(10,3))
			plt.title('Results')
			for i in range(SHOW_NUM):
				a[0][i].imshow(np.reshape(test_x[i], (28,28)))
				a[1][i].imshow(np.reshape(mnist.test.images[i], (28,28)))
				a[2][i].imshow(np.reshape(reconstruction[i], (28,28)))
			plt.show()

	# def get_a_image(self, image_dir):
	# 	image = Image.open(image_dir).convert('L')
	# 	image = image.resize((28, 28), Image.ANTIALIAS)
	# 	image = np.array(image)
	# 	image = tf.cast(image, tf.float32)
	# 	image = tf.reshape(image, [-1, 784])
	# 	print(image.shape)

if __name__ == '__main__':
	AutoEncoder().train()







