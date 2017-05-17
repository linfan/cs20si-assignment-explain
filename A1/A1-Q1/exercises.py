"""
Simple TensorFlow exercises
You should thoroughly test your code
"""

import tensorflow as tf

tf.InteractiveSession()

###############################################################################
# 1a: Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
###############################################################################

x = tf.random_uniform([])  # Empty array as shape creates a scalar.
y = tf.random_uniform([])
out = tf.cond(tf.less(x, y), lambda: tf.add(x, y), lambda: tf.subtract(x, y))
print(out.eval())

###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from -1 and 1.
# Return x + y if x < y, x - y if x > y, 0 otherwise.
# Hint: Look up tf.case().
###############################################################################

# YOUR CODE
x = tf.constant(2, tf.float32)  # tf.random_uniform([], -1, 1, tf.float32)
y = tf.constant(1, tf.float32)  # tf.random_uniform([], -1, 1, tf.float32)
out = tf.case({
    x < y: lambda: x + y,  # tf.less(x, y): lambda: tf.add(x, y)
    x > y: lambda: x - y},  # tf.greater(x, y): lambda: tf.subtract(x, y)
    lambda: tf.constant(0, tf.float32))
print(out.eval())

# Note:
# 文档路径 https://www.tensorflow.org/api_docs/python/tf
# - tf.cond like if-else
# - tf.case like switch-case
# 注意以下比较方法如果参数是Tensor，结果也会返回Tensor，其中每一位都是Boolean值
# - tf.equal(x, y) => if(x == y)
# - tf.less(x, y) => if(x < y)
# - tf.greater(x, y) => if(x > y)

###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]] 
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################

# YOUR CODE
x = tf.constant([[0, -2, -1], [0, 1, 2]])
y = tf.zeros_like(x)
out = tf.equal(x, y)
print(out.eval())

###############################################################################
# 1d: Create the tensor x of value 
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
# Hint: Use tf.where().
# Then extract elements whose values are greater than 30.
# Hint: Use tf.gather().
###############################################################################

# YOUR CODE
x = tf.constant([29.05088806, 27.61298943, 31.19073486, 29.35532951,
                 30.97266006, 26.67541885, 38.08450317, 20.74983215,
                 34.94445419, 34.45999146, 29.06485367, 36.01657104,
                 27.88236427, 20.56035233, 30.20379066, 29.51215172,
                 33.71149445, 28.59134293, 36.05556488, 28.66994858])
index = tf.where(x > 30)
print(index.eval())
out = tf.gather(x, index)
print(out.eval())

# - tf.where 输入一个Boolean矩阵，输出为True的下标列表
# - tf.where 输入一个Boolean矩阵，根据其中每一位置的真值，填入指定两个矩阵中的相应位置数值
# - tf.gather 从给的数据列表中，抽取指定下标位置的值

###############################################################################
# 1e: Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
# Hint: Use tf.range() and tf.diag().
###############################################################################

# YOUR CODE
x = tf.range(1, 7)
out = tf.diag(x)
print(out.eval())

# - tf.diag 用指定列表值生成对角矩阵

###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
# Hint: Look at tf.matrix_determinant().
###############################################################################

# YOUR CODE
x = tf.random_uniform([10, 10], dtype=tf.float32)
out = tf.matrix_determinant(x)
print(out.eval())

###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 09].
# Return the unique elements in x
# Hint: use tf.unique(). Keep in mind that tf.unique() returns a tuple.
###############################################################################

# YOUR CODE
x = [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9]
out, _ = tf.unique(x)
print(out.eval())

# - tf.unique 返回输入值列表中不重复值的列表

###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution.
# Use tf.less() and tf.select() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: see the Huber loss function in the lecture slides 3.
###############################################################################

# YOUR CODE
tf.set_random_seed(0)
x = tf.random_normal([300])
y = tf.random_normal([300])

# >> method-case
with tf.Session() as s:
    out = tf.case({
        tf.less(tf.reduce_mean(x - y), 0): lambda: tf.reduce_mean(tf.square(x - y))
    }, lambda: tf.reduce_sum(tf.abs(x - y)))
    print(s.run(out))

# >> method-cond
with tf.Session() as s:
    out = tf.cond(tf.less(tf.reduce_mean(x - y), 0),
                  lambda: tf.reduce_mean(tf.square(x - y)),
                  lambda: tf.reduce_sum(tf.abs(x - y)))
    print(s.run(out))

# >> method-where
with tf.Session() as s:
    out = tf.where(tf.less(tf.reduce_mean(x - y), 0),
                   tf.reduce_mean(tf.square(x - y)),
                   tf.reduce_sum(tf.abs(x - y)))
    print(s.run(out))

# - tf.reduce_mean 求平均值
# - tf.reduce_sum 求和
# tf.select deprecated - https://github.com/tensorflow/tensorflow/blob/master/RELEASE.md
