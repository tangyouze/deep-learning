import tensorflow as tf
# Sandbox Solution
# Note: You can't run code in this tab

# print(train_features, train_labels)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    # Softmax
    prediction = tf.nn.softmax([0.3, 0.1, 0.2])
    print(prediction)
    r = session.run(prediction)
    print(r)
