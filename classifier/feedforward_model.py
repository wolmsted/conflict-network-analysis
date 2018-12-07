import os
import shutil
import sys
import time
import json
import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from model import Model

features_dir = '../data/'
summaries_dir = '../summaries'
summaries_train_dir = '../summaries/non_text_train'
summaries_dev_dir = '../summaries/non_text_dev'
summaries_loss_dir = '../summaries/non_text_loss'

class Config(object):
    """
    Holds model hyperparams and data information.
    """
    n_features = 315
    n_classes = 9598
    n_layers = 5
    dropout = 0.1
    hidden_sizes = [315, 150, 75, 35, 15]
    batch_size = 512
    n_epochs = 10
    lr = 0.001


class FeedForwardModel(Model):
    """
    Implements a feedforward neural network
    """

    def add_placeholders(self):
        """
        Generates placeholder variables to represent the input tensors
        """
        self.input_placeholder = tf.placeholder(tf.float32, shape=[None, self.config.n_features])
        self.labels_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.n_classes])

    def create_feed_dict(self, inputs_batch, labels_batch):
        """
        Creates the feed_dict for the dependency parser.
        """
        feed_dict = {self.input_placeholder: inputs_batch, self.labels_placeholder: labels_batch}
        return feed_dict

    def add_prediction_op(self):
        """
        Adds the 1-hidden-layer NN:
            h = Relu(xW + b1)
            pred = h + b2
        """
        x = self.input_placeholder
        x = tf.contrib.layers.batch_norm(x, scale=True)
        cache = {}
        for n in range(1, self.config.n_layers):
            cache['W' + str(n)] = tf.get_variable('W' + str(n), shape=[self.config.hidden_sizes[n - 1], self.config.hidden_sizes[n]], initializer=tf.contrib.layers.xavier_initializer())
            cache['b' + str(n)] = tf.get_variable('b' + str(n), shape=[1, self.config.hidden_sizes[n]], initializer=tf.zeros_initializer())

        cache['h1'] = tf.nn.relu(tf.matmul(x, cache['W1']) + cache['b1'])
        for n in range(2, self.config.n_layers):
            cache['h' + str(n)] = tf.nn.relu(tf.matmul(cache['h' + str(n - 1)], cache['W' + str(n)]) + cache['b' + str(n)])
        U = tf.get_variable("U", shape=[self.config.hidden_sizes[-1], self.config.n_classes], initializer=tf.contrib.layers.xavier_initializer())
        b_last = tf.get_variable("b_last", shape=[1, self.config.n_classes], initializer=tf.zeros_initializer())
        pred = tf.matmul(cache['h' + str(self.config.n_layers - 1)], U) + b_last
        return pred



    def add_evaluation_op(self, pred):
        """
        Calculates accuracy from predicted labels
        """
        y_hat = tf.argmax(tf.nn.softmax(pred), 1)
        y = tf.argmax(self.labels_placeholder, 1)
        accuracy = tf.to_float(tf.reduce_sum(tf.cast(tf.equal(y_hat, y), tf.int32))) / tf.to_float(tf.size(y))

        # pos_precision, pos_recall, pos_f1, neg_precision, neg_recall, neg_f1 = self.calculate_metrics(y, y_hat)

        return (y_hat, y, accuracy)

    def add_loss_op(self, pred):
        """
        Adds Ops for the cross entropy loss.
        """
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=pred))
        return loss

    def add_training_op(self, loss):
        """
        Sets up the training Ops.
        """
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        train_op = opt.minimize(loss)
        return train_op

    def evaluate_on_set(self, sess, inputs, labels):
        feed = self.create_feed_dict(inputs, labels_batch=labels)
        data, _ = sess.run([self.evaluate, self.pred], feed_dict=feed)
        return data

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        
        return loss

    def run_epoch(self, sess, train_examples, dev_set, train_writer, dev_writer, loss_writer, epoch):
        dev_inputs = dev_set[:,1:]
        dev_labels = to_categorical(dev_set[:,0], num_classes=9598)
        train_inputs = train_examples[:,1:]
        train_labels = to_categorical(train_examples[:,0], num_classes=9598)
        n_minibatches = 1 + len(train_examples) / self.config.batch_size
        prog = tf.keras.utils.Progbar(target=n_minibatches)
        
        
       
        for i, (train_x, train_y) in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, train_x, train_y)
            
            loss_summary = tf.Summary()
            loss_summary.value.add(tag='loss', simple_value=loss)
            loss_writer.add_summary(loss_summary, epoch * n_minibatches + i)
            loss_writer.flush()

            prog.update(i + 1, [("train loss", loss)], force=i + 1 == n_minibatches)

        print('\nEvaluating on train set',)
        train_acc = self.evaluate_on_set(sess, train_inputs, train_labels)
        print('- train accuracy: {:.4f}'.format(train_acc))

        train_summary = tf.Summary()
        train_summary.value.add(tag='accuracy', simple_value=train_acc)
        train_writer.add_summary(train_summary, epoch)

        

        print("Evaluating on dev set",)
        dev_acc = self.evaluate_on_set(sess, dev_inputs, dev_labels)
        print("- dev accuracy: {:.4f}".format(dev_acc))

        dev_summary = tf.Summary()
        dev_summary.value.add(tag='accuracy', simple_value=dev_acc)
        dev_writer.add_summary(dev_summary, epoch)

        return dev_acc

    def fit(self, sess, saver, train_examples, dev_set, train_writer, dev_writer, loss_writer):
        best_dev_acc = 0.0
        for epoch in range(self.config.n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_acc = self.run_epoch(sess, train_examples, dev_set, train_writer, dev_writer, loss_writer, epoch)
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                if saver:
                    print("New best dev accuracy! Saving model in ../data/feedforward_weights/best.weights")
                    saver.save(sess, '../data/feedforward_weights/best.weights')

            train_writer.flush()
            dev_writer.flush()
            print()

    def __init__(self, config):
        self.config = config
        self.build()

def minibatches(train_examples, batch_size):
    batches = []
    n_batches = int(len(train_examples) / batch_size)
    for i in range(n_batches):
        sample = train_examples[i * batch_size:(i + 1) * batch_size,1:]
        label = to_categorical(train_examples[i * batch_size:(i + 1) * batch_size,0], num_classes=9598)
        batches.append((sample, label))
    sample = train_examples[n_batches * batch_size:,1:]
    label = to_categorical(train_examples[n_batches * batch_size:,0], num_classes=9598)
    batches.append((sample, label))
    return batches

def main():
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    config = Config()

    train_examples = np.genfromtxt('train.csv', delimiter=',')
    dev_set = np.genfromtxt('dev.csv', delimiter=',')
    test_set = np.genfromtxt('test.csv', delimiter=',')

    id_to_group = json.load(open('id_to_group.json'))

    if not os.path.exists('../data/feedforward_weights/'):
        os.makedirs('../data/feedforward_weights/')

    if not os.path.exists(summaries_dir):
        os.makedirs(summaries_dir)

    if os.path.exists(summaries_train_dir):
        shutil.rmtree(summaries_train_dir)
    if os.path.exists(summaries_dev_dir):
        shutil.rmtree(summaries_dev_dir)
    if os.path.exists(summaries_loss_dir):
        shutil.rmtree(summaries_loss_dir)

    os.makedirs(summaries_train_dir)
    os.makedirs(summaries_dev_dir)
    os.makedirs(summaries_loss_dir)

    if not os.path.exists(summaries_train_dir):
        os.makedirs(summaries_train_dir)
    if not os.path.exists(summaries_dev_dir):
        os.makedirs(summaries_dev_dir)

    with tf.Graph().as_default() as graph:
        print("Building model...",)
        start = time.time()
        model = FeedForwardModel(config)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver()
        print("took {:.2f} seconds\n".format(time.time() - start))
    graph.finalize()

    with tf.Session(graph=graph) as session:
        train_writer = tf.summary.FileWriter(summaries_train_dir, session.graph)
        dev_writer = tf.summary.FileWriter(summaries_dev_dir, session.graph)
        loss_writer = tf.summary.FileWriter(summaries_loss_dir, session.graph)


        session.run(init_op)

        # print(80 * "=")
        # print("TRAINING")
        # print(80 * "=")
        # model.fit(session, saver, train_examples, dev_set, train_writer, dev_writer, loss_writer)

        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        saver.restore(session, '../data/feedforward_weights/best.weights')
        print("Final evaluation on test set",)
        test_inputs = test_set[:,1:]
        test_labels = to_categorical(test_set[:,0], num_classes=9598)
        y_hat, y, test_acc = model.evaluate_on_set(session, test_inputs, test_labels)
        for i in range(len(y_hat)):
            print(id_to_group[str(y_hat[-i])], id_to_group[str(y[-i])])
        print('- test accuracy: {:.4f}'.format(test_acc))
        # print('- test pos precision: {:.4f}'.format(test_pos_precision))
        # print('- test pos recall: {:.4f}'.format(test_pos_recall))
        # print('- test pos f1: {:.4f}'.format(test_pos_f1))
        # print('- test neg precision: {:.4f}'.format(test_neg_precision))
        # print('- test neg recall: {:.4f}'.format(test_neg_recall))
        # print('- test neg f1: {:.4f}'.format(test_neg_f1))


if __name__ == '__main__':
    main()
