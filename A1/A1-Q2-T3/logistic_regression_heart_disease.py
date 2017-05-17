"""
Starter code for logistic regression model to solve OCR task 
with Heart Disease Data in TensorFlow
MNIST dataset: yann.lecun.com/exdb/mnist/
"""

import tensorflow as tf
import numpy as np
from random import randint
from random import shuffle
import time

# Constants
DATA_FILE = 'heart.txt'

# Define parameters for the model
learning_rate = 0.01
batch_size = 10
n_epochs = 100
input_len = 9
output_len = 2

# Step 1: Read in data
raw_data = []
for line in open(DATA_FILE):
    raw_data.append(line.strip('\n').split('\t'))

# Pre-processing

# ===== 处理非数值特征 =====

raw_data = raw_data[1:]
for record in raw_data:
    record[4] = int(record[4] == '"Present"')

raw_set = [list(map(float, s)) for s in raw_data]

training_set = raw_set[:400]
testing_set = raw_set[400:]

# ===== 增加正样本数量 =====

# positive_training_set = [s for s in training_set if s[-1] > 0.5]
# negative_training_set = [s for s in training_set if s[-1] < 0.5]
#
# print("添加前：正样本%d, 负样本%d" % (len(negative_training_set), len(positive_training_set)))
# positive_count = len(positive_training_set)
# positive_negative_delta = len(negative_training_set) - positive_count
# for i in range(positive_negative_delta):
#     # 此处必须使用切片，否则append的是一个引用，在归一化时候会出现重复归一化的情况
#     positive_training_set.append(positive_training_set[randint(0, positive_count - 1)][:])
# print("添加后：正样本%d, 负样本%d" % (len(negative_training_set), len(positive_training_set)))
#
# training_set = positive_training_set
# training_set.extend(negative_training_set)
# shuffle(training_set)

# ===== 特征归一化 =====

feature_value_range = []
feature_scale_rate = []

for i in range(9):
    feature_value_range.append([min(training_set, key=lambda x: x[i])[i],
                                max(training_set, key=lambda x: x[i])[i]])
    feature_scale_rate.append(feature_value_range[i][1] - feature_value_range[i][0])
    # print("归一化前特性%d范围：%f ~ %f" % (i, feature_value_range[i][0], feature_value_range[i][1]))

for i in range(9):
    for s in training_set:
        s[i] = (s[i] - feature_value_range[i][0]) / feature_scale_rate[i]
    for s in testing_set:
        s[i] = (s[i] - feature_value_range[i][0]) / feature_scale_rate[i]
    # print("归一化后特性%d范围：%f ~ %f" % (i, min(training_set, key=lambda x: x[i])[i],
    #                               max(training_set, key=lambda x: x[i])[i]))

# ===== 拆分特征和标签 =====

training_features = [s[:9] for s in training_set]
training_labels = [[int(s[-1] < 0.5), int(s[-1] > 0.5)] for s in training_set]

testing_features = [s[:9] for s in testing_set]
testing_labels = [[int(s[-1] < 0.5), int(s[-1] > 0.5)] for s in testing_set]

# Step 2: create placeholders for features and labels
X = tf.placeholder(tf.float32, [batch_size, input_len], name='X_placeholder')
Y = tf.placeholder(tf.float32, [batch_size, output_len], name='Y_placeholder')

# Step 3: create weights and bias
# weights and biases are initialized to 0
# shape of w depends on the dimension of X and Y so that Y = X * w + b
# shape of b depends on Y
w = tf.Variable(tf.zeros([input_len, output_len]), name='weights')  # tf.random_normal(shape=[784, 10], stddev=0.01)
b = tf.Variable(tf.zeros([1, output_len]), name="bias")

# - 维度 [a, b] 是"a行b列" 矩阵
# - 矩阵乘法 [a, b]*[c, d] 可计算，则 b==c，结果维度是 [a, d]

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# to get the probability distribution of possible label of the image
# DO NOT DO SOFTMAX HERE
y = tf.matmul(X, w) + b

# Step 5: define loss function
# use cross entropy loss of the real labels with the softmax of logits
# use the method:
# tf.nn.softmax_cross_entropy_with_logits(logits, Y)
# then use tf.reduce_mean to get the mean loss of the batch
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=Y, name='loss'))

# Step 6: define training op
# using gradient descent to minimize loss
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    # to visualize using TensorBoard
    writer = tf.summary.FileWriter('./graphs', sess.graph)

    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    n_batches = int(len(training_set) / batch_size)
    for i in range(n_epochs):  # train the model n_epochs times
        total_loss = 0

        for n in range(n_batches):
            X_batch = training_features[n*batch_size:(n+1)*batch_size]
            Y_batch = training_labels[n*batch_size:(n+1)*batch_size]

            # TO-DO: run optimizer + fetch loss_batch
            _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})

            total_loss += loss_batch
        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))

    print('Total time: {0} seconds'.format(time.time() - start_time))

    print('Optimization Finished!')  # should be around 0.35 after 25 epochs

    # test the model
    n_batches = int(len(testing_set) / batch_size)
    total_correct_preds = 0
    for n in range(n_batches):
        X_batch = testing_features[n*batch_size:(n+1)*batch_size]
        Y_batch = testing_labels[n*batch_size:(n+1)*batch_size]
        _, loss_batch, logits_batch = sess.run([optimizer, loss, y], feed_dict={X: X_batch, Y: Y_batch})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))  # need numpy.count_nonzero(boolarr) :(
        total_correct_preds += sess.run(accuracy)

    print('Accuracy {0}%'.format(100 * total_correct_preds / (n_batches * batch_size)))
    writer.close()
