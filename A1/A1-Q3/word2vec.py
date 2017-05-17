""" word2vec with NCE loss 
and code to visualize the embeddings on TensorBoard
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from datetime import datetime
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf

from process_data import process_data

VOCAB_SIZE = 1000  # 50000
BATCH_SIZE = 128
EMBED_SIZE = 128  # dimension of the word embedding vectors
SKIP_WINDOW = 1  # the context window
NUM_SAMPLED = 64  # Number of negative examples to sample.
LEARNING_RATE = 0.01
NUM_TRAIN_STEPS = 10000
WEIGHTS_FLD = 'processed/'
SKIP_STEP = 2
USE_PREDICT_INSTEAD_OF_COUNT = False
USE_UPDATE_IN_PLACE_INSTEAD_OF_MATRIX_ADD = True


class SkipGramModel:
    """ Build the graph for word2vec model """

    def __init__(self, vocab_size, embed_size, batch_size, num_sampled, learning_rate):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.lr = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholders(self):
        """ Step 1: define the placeholders for input and output """
        with tf.name_scope("data"):
            self.center_words = tf.placeholder(tf.int32, shape=[self.batch_size], name='center_words')
            self.target_words = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='target_words')

    def _create_embedding(self):
        """ Step 2: define weights. In word2vec, it's actually the weights that we care about """
        # Assemble this part of the graph on the CPU. You can change it to GPU if you have GPU
        with tf.device('/cpu:0'):
            with tf.name_scope("embed"):
                self.embed_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0),
                                                name='embed_matrix')

    def _create_loss(self):
        """ Step 3 + 4: define the model + the loss function """
        with tf.device('/cpu:0'):
            with tf.name_scope("loss"):
                # Step 3: define the inference
                embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embed')

                # Step 4: define loss function
                # construct variables for NCE loss
                nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size],
                                                             stddev=1.0 / (self.embed_size ** 0.5)), name='nce_weight')
                nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')

                # define loss function to be NCE loss function
                self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, biases=nce_bias,
                                                          labels=self.target_words, inputs=embed,
                                                          num_sampled=self.num_sampled, num_classes=self.vocab_size),
                                           name='loss')

    def _create_optimizer(self):
        """ Step 5: define optimizer """
        with tf.device('/cpu:0'):
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,
                                                                                 global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """ Build the graph for our model """
        self._create_placeholders()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()


def save_embed_matrix(sess, embed_matrix):
    # code to visualize the embeddings. uncomment the below to visualize embeddings
    final_embed_matrix = sess.run(embed_matrix)  # 更新词向量矩阵
    # it has to variable. constants don't work here. you can't reuse model.embed_matrix
    embedding_var = tf.Variable(final_embed_matrix[:1000], name='embedding')
    sess.run(embedding_var.initializer)
    config = projector.ProjectorConfig()
    summary_writer = tf.summary.FileWriter(WEIGHTS_FLD)
    # add embedding to the config file
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # link this tensor to its metadata file, in this case the first 500 words of vocab
    embedding.metadata_path = WEIGHTS_FLD + 'vocab_1000.tsv'
    # saves a configuration file that TensorBoard will read during startup.
    projector.visualize_embeddings(summary_writer, config)
    saver_embed = tf.train.Saver([embedding_var])
    saver_embed.save(sess, WEIGHTS_FLD + 'model3.ckpt', 1)  # 单独保存包含词向量的Session数据为Checkpoint


def do_predict(model, batch_gen, num_train_steps):
    saver = tf.train.Saver()  # defaults to saving all variables - in this case embed_matrix, nce_weight, nce_bias
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        # if that checkpoint exists, restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        total_loss = 0.0  # we use this to calculate late average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('sg_graphs/' + str(LEARNING_RATE), sess.graph)  # 记录数据图表的Writer
        initial_step = model.global_step.eval()  # 避免从Checkpoint读入时初始值不是0
        for index in range(initial_step, initial_step + num_train_steps):  # 每个迭代总是训练num_train_steps次
            centers, targets = next(batch_gen)  # 获得下一组的中间词(center)和附近范围内的其中一个词(target)组成的batch
            feed_dict = {model.center_words: centers, model.target_words: targets}  # 把这个batch填入place_holder
            loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op],
                                              feed_dict=feed_dict)  # 计算
            writer.add_summary(summary, global_step=index)  # 写入所有的指标数据
            total_loss += loss_batch  # 累加这次迭代的Lost到总Lost
            if (index + 1) % SKIP_STEP == 0:  # 每当达到了一个检查周期
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))  # 打印这个周期的平均Lost
                total_loss = 0.0  # 将总Lost归零
                saver.save(sess, 'checkpoints/skip-gram', index)  # 保存当前计算状态为Checkpoint
                save_embed_matrix(sess, model.embed_matrix)


def do_count(batch_gen, num_train_steps):
    with tf.Session() as sess:
        co_occurrence_matrix = tf.Variable(tf.zeros([VOCAB_SIZE, VOCAB_SIZE], dtype=tf.int32),
                                           name='co_occurrence_matrix')
        sess.run(tf.global_variables_initializer())

        if USE_UPDATE_IN_PLACE_INSTEAD_OF_MATRIX_ADD:  # 方法一：直接原地更新
            for step in range(num_train_steps):
                centers, targets = next(batch_gen)
                for index in range(BATCH_SIZE):
                    x = int(centers[index])
                    y = int(targets[index][0])
                    row = tf.gather(co_occurrence_matrix, x)
                    new_row = tf.concat([row[:y], [co_occurrence_matrix[x][y]], row[y+1:]], axis=0)
                    co_occurrence_matrix.assign(tf.scatter_update(co_occurrence_matrix, x, new_row))
                if (step + 1) % SKIP_STEP == 0:
                    do_pca(co_occurrence_matrix, sess, step)

        else:  # 方法二：使用矩阵加法
            for step in range(num_train_steps):
                centers, targets = next(batch_gen)
                for index in range(BATCH_SIZE):
                    pos = [int(centers[index]), int(targets[index][0])]
                    pos_matrix = tf.sparse_tensor_to_dense(
                        tf.SparseTensor(indices=[pos], values=[1], dense_shape=[VOCAB_SIZE, VOCAB_SIZE]))
                    co_occurrence_matrix.assign_add(pos_matrix)
                if (step + 1) % SKIP_STEP == 0:
                    do_pca(co_occurrence_matrix, sess, step)


def do_pca(co_occurrence_matrix, sess, step):
    s, u, v = tf.svd(tf.cast(co_occurrence_matrix, tf.float32))
    embed_matrix = tf.matmul(u[:, :EMBED_SIZE], tf.diag(s[:EMBED_SIZE]))
    save_embed_matrix(sess, embed_matrix)
    print('%s => Step %s' % (datetime.now(), step))


def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    if USE_PREDICT_INSTEAD_OF_COUNT:
        model = SkipGramModel(VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
        model.build_graph()
        do_predict(model, batch_gen, NUM_TRAIN_STEPS)
    else:
        do_count(batch_gen, NUM_TRAIN_STEPS)


if __name__ == '__main__':
    main()
