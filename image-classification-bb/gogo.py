import tensorflow as tf

hello = tf.constant('hello')
world = tf.constant('world')
x = tf.placeholder(tf.int32)
x0 = tf.placeholder(tf.int32)
x1 = tf.placeholder(tf.float32)
x3 = tf.cast(x0, tf.float32) - x1
z = x + x
print(z)
print(tf.add(x, x))
with tf.Session() as sess:
    h = hello + '  ' + world + str(x)
    output = sess.run(x + x, feed_dict={x: 123})
    print(output)
print('-------------------variable-------------------')

var = tf.Variable(tf.truncated_normal((120, 5)))
print(tf.zeros(2))
print(var)
init = tf.global_variables_initializer()
print(init)
with tf.Session() as sess:
    o = sess.run(init)
    print(o)
