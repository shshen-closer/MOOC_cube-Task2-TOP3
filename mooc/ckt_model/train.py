# -*- coding:utf-8 -*-
__author__ = 'shshen'

import os
import sys
import time
import logging
import random
import pandas as pd
import tensorflow as tf
from datetime import datetime
import numpy as np
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import metrics
from model_ckt import CKT
#from model_LSTM import CKT
import checkmate as cm
import data_helpers as dh


# Parameters
# ==================================================

TRAIN_OR_RESTORE = 'T' #input("Train or Restore?(T/R): ")

while not (TRAIN_OR_RESTORE.isalpha() and TRAIN_OR_RESTORE.upper() in ['T', 'R']):
    TRAIN_OR_RESTORE = input("The format of your input is illegal, please re-input: ")
logging.info("The format of your input is legal, now loading to next step...")

TRAIN_OR_RESTORE = TRAIN_OR_RESTORE.upper()

if TRAIN_OR_RESTORE == 'T':
    logger = dh.logger_fn("tflog", "logs/training_kfold_{0}_{1}_time_{2}.log".format(sys.argv[1], sys.argv[0], int(time.time())))
if TRAIN_OR_RESTORE == 'R':
    logger = dh.logger_fn("tflog", "logs/restore-{0}.log".format(time.asctime()).replace(':', '_'))

kfold= int(sys.argv[1])
batch_size = int(sys.argv[2])
tf.compat.v1.flags.DEFINE_string("train_or_restore", TRAIN_OR_RESTORE, "Train or Restore.")
tf.compat.v1.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.compat.v1.flags.DEFINE_float("norm_ratio", 5, "The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
tf.compat.v1.flags.DEFINE_float("keep_prob", 0.5, "Keep probability for dropout")
tf.compat.v1.flags.DEFINE_float("radio", 0.6, "split radio")
tf.compat.v1.flags.DEFINE_integer("hidden_size", 256, "The number of hidden nodes (Integer)")
tf.compat.v1.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.compat.v1.flags.DEFINE_integer("batch_size", batch_size , "Batch size for training.")
tf.compat.v1.flags.DEFINE_integer("epochs", 3, "Number of epochs to train for.")
tf.compat.v1.flags.DEFINE_integer("kfold", kfold, "Number of epochs to train for.")

tf.compat.v1.flags.DEFINE_integer("decay_steps",1, "how many steps before decay learning rate. (default: 500)")
tf.compat.v1.flags.DEFINE_float("decay_rate", 0.5, "Rate of decay for learning rate. (default: 0.95)")
tf.compat.v1.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many steps (default: 1000)")
tf.compat.v1.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 50)")

# Misc Parameters
tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.compat.v1.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.compat.v1.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100



def train():
    """Training model."""

    # Load sentences, labels, and training parameters
    logger.info("Loading data...")

    logger.info("Training data processing...")

    train_q = np.load("../data/data_30/train_q_" + str(kfold) + ".npy", allow_pickle=True)
    train_a = np.load("../data/data_30/train_a_" + str(kfold) + ".npy", allow_pickle=True)
    train_len = np.load("../data/data_30/train_length_" + str(kfold) + ".npy", allow_pickle=True)
    valid_q = np.load("../data/data_30/valid_q_" + str(kfold) + ".npy", allow_pickle=True)
    valid_a = np.load("../data/data_30/valid_a_" + str(kfold) + ".npy", allow_pickle=True)
    valid_len = np.load("../data/data_30/valid_length_" + str(kfold) + ".npy", allow_pickle=True)


    with open('../data/problem2kpoint', 'r', encoding='utf8') as fi:
        for line in fi:
            problem2kpoint = eval(line)

    print(len(train_q))
  #  train_q = train_q[:10000]
  #  train_a = train_a[:10000]
   # train_sid= train_sid[:int(len(train_sid)/2)]
  #  valid_q = valid_q[:10000]
   # valid_a = valid_a[:10000]
   # valid_sid = valid_sid[:int(len(valid_sid)/2)]
    #logger.info("complete")
    # Build a graph and lstm_3 object
    with tf.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            ckt = CKT(
                batch_size = FLAGS.batch_size,
                num_skills = 74,
                hidden_size = FLAGS.hidden_size,
                )


            # Define training procedure
            with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
                learning_rate = tf.compat.v1.train.exponential_decay(learning_rate=FLAGS.learning_rate,
                                                           global_step=ckt.global_step, decay_steps=(len(train_q)//FLAGS.batch_size +1) * FLAGS.decay_steps,
                                                           decay_rate=FLAGS.decay_rate, staircase=True)
               # learning_rate = tf.train.piecewise_constant(FLAGS.epochs, boundaries=[7,10], values=[0.005, 0.0005, 0.0001])
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
                #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
                #grads, vars = zip(*optimizer.compute_gradients(ckt.loss))
                #grads, _ = tf.clip_by_global_norm(grads, clip_norm=FLAGS.norm_ratio)
                #train_op = optimizer.apply_gradients(zip(grads, vars), global_step=ckt.global_step, name="train_op")
                train_op = optimizer.minimize(ckt.loss, global_step=ckt.global_step, name="train_op")

            # Output directory for models and summaries
            if FLAGS.train_or_restore == 'R':
                MODEL = input("Please input the checkpoints model you want to restore, "
                              "it should be like(990175368): ")  # The model you want to restore

                while not (MODEL.isdigit() and len(MODEL) == 10):
                    MODEL = input("The format of your input is illegal, please re-input: ")
                logger.info("The format of your input is legal, now loading to next step...")
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
                logger.info("Writing to {0}\n".format(out_dir))
            else:
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                logger.info("Writing to {0}\n".format(out_dir))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            best_checkpoint_dir = os.path.abspath(os.path.join(out_dir, "bestcheckpoints"))

            # Summaries for loss
            loss_summary = tf.compat.v1.summary.scalar("loss", ckt.loss)

            # Train summaries
            train_summary_op = tf.compat.v1.summary.merge([loss_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.compat.v1.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.compat.v1.summary.merge([loss_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.compat.v1.summary.FileWriter(validation_summary_dir, sess.graph)

            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            best_saver = cm.BestCheckpointSaver(save_dir=best_checkpoint_dir, num_to_keep=3, maximize=True)

            if FLAGS.train_or_restore == 'R':
                # Load ckt model
                logger.info("Loading model...")
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                logger.info(checkpoint_file)

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            else:
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(tf.compat.v1.global_variables_initializer())
                sess.run(tf.compat.v1.local_variables_initializer())



            current_step = sess.run(ckt.global_step)

            def train_step(input_id, x_answer, x_answer1,valid_id, target_correctness, target_correctness1, length):
                """A single training step"""

                #print(ability)
                feed_dict = {
                    ckt.input_id:input_id,
                    ckt.x_answer: x_answer,
                    ckt.x_answer1: x_answer1,
                    ckt.valid_id: valid_id,
                    ckt.target_correctness: target_correctness,
                    ckt.target_correctness1: target_correctness1,
                    ckt.seq_length: length,
                    ckt.dropout_keep_prob: FLAGS.keep_prob,
                    ckt.is_training: True
                }
                _, step, summaries, pred, loss = sess.run(
                    [train_op, ckt.global_step, train_summary_op, ckt.pred, ckt.loss], feed_dict)

                
                acc = np.mean(np.equal(np.reshape(target_correctness, [-1]), np.argmax(pred, axis = -1)).astype(int))
                logger.info("step {0}: loss {1:g} acc:{2:g} ".format(step,loss, acc))
                train_summary_writer.add_summary(summaries, step)
                return pred

            def validation_step( input_id, x_answer,  x_answer1,valid_id, target_correctness, target_correctness1, length):
                """Evaluates model on a validation set"""

                feed_dict = {
                    ckt.input_id:input_id,
                    ckt.x_answer: x_answer,
                    ckt.x_answer1: x_answer1,
                    ckt.valid_id: valid_id,
                    ckt.target_correctness: target_correctness,
                    ckt.target_correctness1: target_correctness1,
                    ckt.seq_length: length,
                    ckt.dropout_keep_prob: 0.0,
                    ckt.is_training: False
                }
                step, summaries, pred, loss = sess.run(
                    [ckt.global_step, validation_summary_op, ckt.pred, ckt.loss], feed_dict)
                validation_summary_writer.add_summary(summaries, step)
                return pred
            # Training loop. For each batch...

            def one_hot_answer(input_data,length):
                output = np.zeros((length,30,2),  dtype=float)
                for item in range(len(input_data)):
                    for iii in range(len(input_data[item])):
                        if input_data[item][iii] != 2:
                            output[item,iii,input_data[item][iii]] = 1
                return output


            run_time = []
            m_acc = 0
            for iii in range(FLAGS.epochs):
                
                a=datetime.now()
                data_size = len(train_q)
                index = FLAGS.batch_size
                batch = 0
                actual_labels = []
                pred_labels = []
                while(index*batch+FLAGS.batch_size < data_size):
                    #for repeat in range(2):
                    question_id = train_q[index*batch : index*(batch+1)]
                    answer = train_a[index*batch : index*(batch+1)]
                    answer1 = one_hot_answer(answer, FLAGS.batch_size)

                    length = train_len[index*batch : index*(batch+1)]

                    input_id = question_id[:,:29]
                    valid_id = question_id[:, -1]

                    x_answer = answer[:,:29]
                    x_answer1 = answer1[:,:29, :]

                    target_correctness = answer[:,-1]
                    target_correctness1 = answer1[:,-1,:]
                    
                    actual_labels.extend(np.reshape(target_correctness, [-1]))

                    pred = train_step(input_id, x_answer,  x_answer1, valid_id, target_correctness, target_correctness1, length)
                    pred_labels.extend(np.reshape(np.argmax(pred, -1), [-1]))

                    current_step = tf.compat.v1.train.global_step(sess, ckt.global_step)
                    batch += 1

                b=datetime.now()
                e_time = (b-a).total_seconds()
                run_time.append(e_time)

                pred_score = np.equal(actual_labels, pred_labels)
                acc = np.mean(pred_score.astype(int))
                logger.info("epochs {0}: acc {1:g}  ".format((iii +1), acc))

                if((iii+1) % FLAGS.evaluation_interval == 0):
                    logger.info("\nEvaluation:")

                    index = FLAGS.batch_size
                    data_size = len(valid_q)
                    batch = 0
                    actual_labels = []
                    pred_labels = []
                    while(index*batch+FLAGS.batch_size < data_size):
                        question_id = valid_q[index*batch : index*(batch+1)]
                        answer = valid_a[index*batch : index*(batch+1)]

                        answer1 = one_hot_answer(answer, FLAGS.batch_size)
                        
                        length = valid_len[index*batch : index*(batch+1)]
                        input_id = question_id[:,:29]
                        valid_id = question_id[:, -1]
                        x_answer1 = answer1[:,:29,:]
                        x_answer = answer[:,:29]

                        target_correctness1 = answer1[:,-1,:]
                        target_correctness = answer[:,-1]
                        actual_labels.extend(np.reshape(target_correctness, [FLAGS.batch_size]))

                        batch += 1
                        #print(ability)
                        pred = validation_step(input_id, x_answer, x_answer1,valid_id, target_correctness, target_correctness1, length)
                        pred_labels.extend(np.reshape(np.argmax(pred, -1), [-1]))

                        current_step = tf.compat.v1.train.global_step(sess, ckt.global_step)


                    pred_score = np.equal(actual_labels, pred_labels)
                    acc = np.mean(pred_score.astype(int))

                    logger.info("VALIDATION {0}: acc {1:g} ".format((iii +1)/FLAGS.evaluation_interval,acc))

                    if acc > m_acc:
                        m_acc = acc

                    best_saver.handle(acc, sess, current_step)
                if ((iii+1) % FLAGS.checkpoint_every == 0):
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("Saved model checkpoint to {0}\n".format(path))

                logger.info("Epoch {0} has finished!".format(iii + 1))

            logger.info("running time analysis: epoch{0}, avg_time{1}".format(len(run_time), np.mean(run_time)))
            logger.info("max: acc{0:g} ".format(m_acc))
            with open('results.txt', 'a') as fi:
                fi.write( ':\n')
                fi.write("max: acc {0:g}  ".format(m_acc))
                fi.write('\n')
    logger.info("Done.")


if __name__ == '__main__':
    train()
