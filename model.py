import tensorflow as tf
import numpy as np
import math
class ConvKB(object):

    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, sequence_length, num_classes, embedding_size, embedding_des_size, filter_sizes, num_filters, vocab_size,
                 pre_trained=[], pre_trained_des=[], l2_reg_lambda=0.001, batch_size=256, is_trainable=True, useConstantInit=False):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [batch_size, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope("embedding"):
            if pre_trained == []:
                self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -math.sqrt(1.0/embedding_size), math.sqrt(1.0/embedding_size), seed=1234), name="W")
            else:
                self.W = tf.get_variable(name="W2", initializer=pre_trained, trainable=is_trainable)

            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            if pre_trained_des == []:
                self.W_2 = tf.Variable(tf.random_uniform([vocab_size, embedding_des_size], -math.sqrt(1.0/embedding_des_size), math.sqrt(1.0/embedding_des_size), seed=1234), name="W_2")
            else:
                self.W_2 = tf.get_variable(name="W2_2", initializer=pre_trained_des, trainable=is_trainable)

            self.embedded_chars2 = tf.nn.embedding_lookup(self.W_2, self.input_x)
            self.embedded_chars_expanded2 = tf.expand_dims(self.embedded_chars2, -1)            

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []

        
        for i, filter_size in enumerate(filter_sizes):
            print("running with convolutions")
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                if useConstantInit == False:
                    filter_shape = [sequence_length, filter_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1, seed=1234), name="W")
                else:
                    init1 = tf.constant([[[[0.1]]], [[[0.1]]], [[[-0.1]]]])
                    weight_init = tf.tile(init1, [1, filter_size, 1, num_filters])
                    W = tf.get_variable(name="W3", initializer=weight_init)

                b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled_outputs.append(h)
                print("pooled_output.shape : " + str(pooled_outputs)) #=256,1,141,500


        #pooled_outputs_des = []
        for i, filter_size in enumerate(filter_sizes):
            print("running with convolutions")
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                if useConstantInit == False:
                    filter_shape = [sequence_length, filter_size, 1, num_filters]
                    W_2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1, seed=1234), name="W_2")
                else:
                    init1 = tf.constant([[[[0.1]]], [[[0.1]]], [[[-0.1]]]])
                    weight_init = tf.tile(init1, [1, filter_size, 1, num_filters])
                    W_2 = tf.get_variable(name="W3_2", initializer=weight_init)

                b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded2,
                    W_2,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled_outputs.append(h)
                print("pooled_outputs.shape : " + str(pooled_outputs)) #=256,1,3996,500



        # Combine all the pooled features
        self.h_pool = tf.concat(pooled_outputs, 2) 
        print("h_pool.shape : " + str(self.h_pool.shape)) #256,1,50,500
        total_dims = (embedding_size * len(filter_sizes) - sum(filter_sizes) + len(filter_sizes)) * num_filters
        total_des_dims = (embedding_des_size * len(filter_sizes) - sum(filter_sizes) + len(filter_sizes)) * num_filters
        dimension = total_dims + total_des_dims
        print("total_dims : " + str(total_dims)) #=25000
        print("embedding_size = " +str(embedding_size)) #=50
        print("len(filter_sizes) = " +str(len(filter_sizes))) #=100
        print("sum(filter_sizes) = " +str(sum(filter_sizes))) #=1
        print("num_filters = " +str(num_filters)) #=500

        
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, dimension]) #is this input of rnn?
        

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob) #need drop out?

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[dimension, num_classes],
                initializer=tf.contrib.layers.xavier_initializer(seed=1234))
            b = tf.Variable(tf.constant(0.0, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.nn.sigmoid(self.scores)
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softplus(self.scores * self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)