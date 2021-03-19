# -*- coding:utf-8 -*-
__author__ = 'shshen'

import os
import sys
import time
import numpy as np 
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import checkmate as cmm
import data_helpers as dh
import json

# Parameters
# ==================================================
#seq_len= int(sys.argv[1])
#batch_size = int(sys.argv[2])
logger = dh.logger_fn("tflog", "logs/test-{0}.log".format(time.asctime()).replace(':', '_'))
number = sys.argv[1]
file_name = sys.argv[2]
filerun = sys.argv[3]

MODEL = file_name
while not (MODEL.isdigit() and len(MODEL) == 10):
    MODEL = input("The format of your input is illegal, it should be like(90175368), please re-input: ")
logger.info("The format of your input is legal, now loading to next step...")



TESTSET_DIR = 'data/assist2009_updated_all.csv'
MODEL_DIR = filerun  + '/' + MODEL + '/checkpoints/'
BEST_MODEL_DIR =  filerun  + '/' + MODEL + '/bestcheckpoints/'
SAVE_DIR = 'results/' + MODEL

# Data Parameters
tf.compat.v1.flags.DEFINE_string("test_data_file", TESTSET_DIR, "Data source for the test data")
tf.compat.v1.flags.DEFINE_string("checkpoint_dir", MODEL_DIR, "Checkpoint directory from training run")
tf.compat.v1.flags.DEFINE_string("best_checkpoint_dir", BEST_MODEL_DIR, "Best checkpoint directory from training run")

# Model Hyperparameters
tf.compat.v1.flags.DEFINE_float("l2_lambda", 0.0001, "Lambda for l2 loss.")
tf.compat.v1.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.compat.v1.flags.DEFINE_float("norm_ratio", 5, "The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
tf.compat.v1.flags.DEFINE_float("keep_prob", 0.2, "Keep probability for dropout")
tf.compat.v1.flags.DEFINE_float("radio", 0.8, "split radio")
tf.compat.v1.flags.DEFINE_integer("hidden_size", 256, "The number of hidden nodes (Integer)")
tf.compat.v1.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.compat.v1.flags.DEFINE_integer("batch_size", 100 , "Batch size for training.")
tf.compat.v1.flags.DEFINE_integer("epochs", 25, "Number of epochs to train for.")
tf.compat.v1.flags.DEFINE_integer("seq_len", 50, "Number of epochs to train for.")

# Misc Parameters
tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.compat.v1.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.compat.v1.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100
#logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), FLAGS.__getattr__(attr))
   #                             for attr in sorted(FLAGS.__dict__['__wrapped'])], dilim]))
#


    

def test():

    # Load data
    logger.info("Loading data...")

    logger.info("Training data processing...")
    max_num_skills, max_num_steps = 256, 30
    test_q = np.load("../data/data_30/test_q.npy", allow_pickle=True)
    test_a = np.load("../data/data_30/test_a.npy", allow_pickle=True)
    test_len = np.load("../data/data_30/test_length.npy", allow_pickle=True)
    test_id = np.load("../data/data_30/test_id.npy", allow_pickle=True)

    with open('../data/problem2kpoint', 'r', encoding='utf8') as fi:
        for line in fi:
            problem2kpoint = eval(line)

    BEST_OR_LATEST = 'B'

    while not (BEST_OR_LATEST.isalpha() and BEST_OR_LATEST.upper() in ['B', 'L']):
        BEST_OR_LATEST = input("he format of your input is illegal, please re-input: ")
    if BEST_OR_LATEST == 'B':
        logger.info("Loading best model...")
        checkpoint_file = cmm.get_best_checkpoint(FLAGS.best_checkpoint_dir, select_maximum_value=True)
    if BEST_OR_LATEST == 'L':
        logger.info("latest")
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    logger.info(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.compat.v1.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_id = graph.get_operation_by_name("input_id").outputs[0]
            x_answer = graph.get_operation_by_name("x_answer").outputs[0]
            x_answer1 = graph.get_operation_by_name("x_answer1").outputs[0]
            valid_id = graph.get_operation_by_name("valid_id").outputs[0]
            target_correctness = graph.get_operation_by_name("target_correctness").outputs[0]
            target_correctness1 = graph.get_operation_by_name("target_correctness1").outputs[0]
            seq_length = graph.get_operation_by_name("seq_length").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]


            pred = graph.get_operation_by_name("pred").outputs[0]
            
            
            def one_hot_answer(input_data,length):
                output = np.zeros((length,30,2),  dtype=float)
                for item in range(len(input_data)):
                    for iii in range(len(input_data[item])):
                        if input_data[item][iii] != 2:
                            output[item,iii,input_data[item][iii]] = 1
                return output

            
            data_size = len(test_q)
            index = FLAGS.batch_size
            batch = 0

            pred_labels = {}
            while(index*batch < data_size):


                question_id = test_q[index*batch : index*(batch+1)]
                answer = test_a[index*batch : index*(batch+1)]
                answer1 = one_hot_answer(answer, len(answer))

                iid = test_id[index*batch : index*(batch+1)]
                length = test_len[index*batch : index*(batch+1)]
                input_id_b = question_id[:,:29]
                valid_id_b = question_id[:, -1]
                x_answer_b = answer[:,:29]
                x_answer1_b = answer1[:,:29, :]

                target_correctness_b = answer[:,-1]
                target_correctness1_b = answer1[:,-1,:]


                batch += 1

                feed_dict = {
                    input_id:input_id_b,
                    x_answer: x_answer_b,
                    x_answer1: x_answer1_b,
                    valid_id: valid_id_b,
                    target_correctness: target_correctness_b,
                    target_correctness1: target_correctness1_b,
                    seq_length: length,
                    dropout_keep_prob: 0.0,
                    is_training: False
                }
                
                pred_b = sess.run(pred, feed_dict)
                pred_b = np.argmax(pred_b, axis = -1)
                for ps, ids in zip(pred_b, iid):
                    pred_labels[ids] = ps
            print(np.shape(pred_labels))
            sum_s = 0
            for key in pred_labels.keys():
                sum_s += pred_labels[key]
            print(sum_s / len(pred_labels))
            with open('result/pred_labels_' + filerun + number, 'w', encoding='utf8') as fo:
                fo.write(str(pred_labels))

                
                
            

    logger.info("Done.")


if __name__ == '__main__':
    test()
