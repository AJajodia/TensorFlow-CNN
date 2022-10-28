# TODO: Add imports.
import tensorflow as tf
import matplotlib.pyplot as plt

# TODO: Clear the tensorflow graph
tf.reset_default_graph()

test_constant = tf.constant(10.0, dtype=tf.float32)
add_one_operation = test_constant + 1

# TODO: Create placeholders
tf.reset_default_graph()

input_data = tf.placeholder(dtype=tf.float32, shape=None)

double_operation = input_data * 2

# TODO: Create variables
tf.reset_default_graph()

input_data = tf.placeholder(dtype=tf.float32, shape=None)
output_data = tf.placeholder(dtype=tf.float32, shape=None)

slope = tf.Variable(0.5, dtype=tf.float32)
intercept = tf.Variable(3, dtype=tf.float32)

model_operation = slope * input_data + intercept

error = model_operation - output_data
squared_error = tf.square(error)
loss = tf.reduce_mean(squared_error)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
train = optimizer.minimize(loss)

# TODO: Run a session
init = tf.global_variables_initializer()

x_values = [0, 1, 2, 3, 4, 2.5]
y_values = [1, 3, 5, 7, 9, 7]

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        sess.run(train, feed_dict={input_data: x_values, output_data: y_values})
        if i % 100 == 0:
            print(sess.run([slope, intercept]))
            plt.plot(x_values, sess.run(model_operation, feed_dict={input_data: x_values}))

    print(sess.run(loss, feed_dict={input_data: x_values, output_data: y_values}))
    plt.plot(x_values, y_values, 'ro', 'Training Data')
    plt.plot(x_values, sess.run(model_operation, feed_dict={input_data: x_values}))

    plt.show()