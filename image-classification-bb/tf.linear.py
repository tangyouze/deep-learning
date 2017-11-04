import tensorflow as tf

v = tf.Variable(5.2, name='weight')
x = tf.placeholder(dtype=tf.float32)

loss = tf.pow(tf.subtract(v, x), 2)
print('loss', loss)
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('v', v)
    for _ in range(30):
        for possible in [2, 3]:
            y = sess.run(optimizer, feed_dict={x: possible})
            print(y)
            l = sess.run(loss, feed_dict={x: possible})
            print(l)
