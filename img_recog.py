# Tensorflow and numpy to create the neural network
import tensorflow as tf
import numpy as np

# Matplotlib to plot info to show our results
import matplotlib.pyplot as plt

# OS to load files and save checkpoints
import os

import keras
'exec(%matplotlib inline)'

# Load MNIST data from tf examples

image_height = 28
image_width = 28

color_channels = 1

model_name = "mnist"

mnist = tf.contrib.learn.datasets.load_dataset("mnist")

train_data = mnist.train.images
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

eval_data = mnist.test.images
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

category_names = list(map(str, range(10)))

# TODO: Process mnist data

print(train_data.shape)

train_data = np.reshape(train_data, (-1, image_height, image_width, color_channels))

print(train_data.shape)

eval_data = np.reshape(eval_data, (-1, image_height, image_width, color_channels))

# TODO: Load cifar data from file

# # image_height = 32
# image_width = 32
#
# color_channels = 3
#
# model_name = "cifar"
#
#
# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
#
#
# cifar_path = "/home/idstudent/PycharmProjects/AnuJ/cifar-10-batches-py/"
#
# train_data = np.array([])
# train_labels = np.array([])
#
# # Load all the data batches.
# for i in range(1,6):
#     data_batch = unpickle(cifar_path + 'data_batch_' + str(i))
#     train_data = np.append(train_data, data_batch[b'data'])
#     train_labels = np.append(train_labels, data_batch[b'labels'])
#
#
# # Load the eval batch.
# eval_batch = unpickle(cifar_path + 'test_batch')
#
# eval_data = eval_batch[b'data']
# eval_labels = eval_batch[b'labels']
#
# # Load the english category names.
# category_names_bytes = unpickle(cifar_path + 'batches.meta')[b'label_names']
# category_names = list(map(lambda x: x.decode("utf-8"), category_names_bytes))
#
# # TODO: Process Cifar data


def process_data(data):
    float_data = np.array(data, dtype=float) / 255.0

    reshaped_data = np.reshape(float_data, (-1, color_channels, image_height, image_width))

    transposed_data = np.transpose(reshaped_data, [0, 2, 3, 1])

    return transposed_data


train_data = process_data(train_data)

eval_data = process_data(eval_data)

plt.show()

# TODO: The neural network


class ConvNet:
    def __init__(self, image_height, image_width, channels, num_classes):

        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, channels], name="inputs")
        print(self.input_layer.shape)
# Creates placeholder for image input
        conv_layer_1 = tf.layers.conv2d(self.input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        print(conv_layer_1.shape)
# Creates a convolutional layer using ReLu activation (returns value if true, nothing if false)
        pooling_layer_1 = tf.layers.max_pooling2d(conv_layer_1, pool_size=[2, 2], strides=2)
        print(pooling_layer_1.shape)
# Shrinks data for easier manipulation
        conv_layer_2 = tf.layers.conv2d(pooling_layer_1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        print(conv_layer_2.shape)
# Creates second conv2d layer
        pooling_layer_2 = tf.layers.max_pooling2d(conv_layer_2, pool_size=[2, 2], strides=2)
        print(pooling_layer_2.shape)
# Creates second shrinker
        flattened_pooling = tf.layers.flatten(pooling_layer_2)
        dense_layer = tf.layers.dense(flattened_pooling, 124, activation=tf.nn.relu)
        print(dense_layer.shape)
# Flattens 2d image filters to connect layers
        dropout = tf.layers.dropout(dense_layer, rate=0.4)
# Forces some neurons to stop to force more of the network to learn the task
        outputs = tf.layers.dense(dropout, num_classes)
        print(outputs.shape)
# Computes outputs
        self.choice = tf.argmax(outputs, axis=1)
        self.probabilities = tf.nn.softmax(outputs)
# Makes class predictions and get probabilities guessed per class
        self.labels = tf.placeholder(dtype=tf.float32, name="labels")
        self.accuracy, self.accuracy_op = tf.metrics.accuracy(self.labels, self.choice)
# Allows you to feed values to give answers for the network to grade against and creates an operation to update accuracy and a variable to store that value
        one_hot_labels = tf.one_hot(indices=tf.cast(self.labels, dtype=tf.int32), depth=num_classes)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=outputs)
# Creates one-hot labels for outputting integers
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
        self.train_operation = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())

# TODO: initialize variables

training_steps = 20000
batch_size = 32

path = "./" + model_name + "-cnn/"

load_checkpoint = True

performance_graph = np.array([])

# TODO: implement the training loop

tf.reset_default_graph()

dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
dataset = dataset.shuffle(buffer_size=train_labels.shape[0])
dataset = dataset.batch(batch_size)
dataset = dataset.repeat()
# Condition dataset by first building dataset, then shuffling, determining batch size, and repeating

dataset_iterator = dataset.make_initializable_iterator()
next_element = dataset_iterator.get_next()
# Iterates through data

cnn = ConvNet(image_height, image_width, color_channels, 10)
# Actually creates tensorflow graph and cnn

saver = tf.train.Saver(max_to_keep=2)
if not os.path.exists(path):
    os.makedirs(path)
# Saves checkpoints to a directory


with tf.Session() as sess:
    if load_checkpoint:
        checkpoint= tf.train.get_checkpoint_state(path)
        saver.restore(sess, checkpoint.model_checkpoint_path)

    else:
        sess.run(tf.global_variables_initializer())
# Sets up initial weights
    sess.run(tf.local_variables_initializer())
    sess.run(dataset_iterator.initializer)
# Starts variables and dataset iterator
    for step in range(training_steps):
        current_batch = sess.run(next_element)


        batch_inputs = current_batch[0]
        batch_labels = current_batch[1]

        sess.run((cnn.train_operation, cnn.accuracy_op), feed_dict={cnn.input_layer:batch_inputs, cnn.labels:batch_labels})

        if step % 10 == 0:
            performance_graph = np.append(performance_graph, sess.run(cnn.accuracy))

        if step % 100 == 0 and step > 0:
            current_acc = sess.run(cnn.accuracy)
            print(("Accuracy at step " + str(step)) + ": " + str(current_acc))
            print("Saving checkpoint")
            saver.save(sess, path + model_name, step)
    print("Saving final checkpoint for training session.")
    saver.save(sess, path + model_name, step)
# TODO: Display graph of performance over time

plt.plot(performance_graph)
plt.figure().set_facecolor('white')
plt.xlabel("Steps")
plt.ylabel("Accuracy")

plt.show()

















