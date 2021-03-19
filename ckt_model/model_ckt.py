# -*- coding:utf-8 -*-
__author__ = 'shshen'

import numpy as np
import tensorflow as tf

from model_function import *
#from densenet import *
#from resnet import *
class CKT(object):

    def __init__(self, batch_size, num_skills, hidden_size):
        
        self.batch_size = batch_size = batch_size
        self.hidden_size  = hidden_size
        self.num_skills =  num_skills


        self.input_id = tf.compat.v1.placeholder(tf.int32, [None, 29], name="input_id")
        self.x_answer = tf.compat.v1.placeholder(tf.int32, [None, 29], name="x_answer")
        self.x_answer1 = tf.compat.v1.placeholder(tf.float32, [None, 29,2], name="x_answer1")
        self.valid_id = tf.compat.v1.placeholder(tf.int32, [None,], name="valid_id")
        self.target_correctness = tf.compat.v1.placeholder(tf.float32, [None], name="target_correctness")
        self.target_correctness1 = tf.compat.v1.placeholder(tf.float32, [None,2], name="target_correctness1")
        self.seq_length = tf.compat.v1.placeholder(tf.int32, [None,], name="seq_length")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.compat.v1.placeholder(tf.bool, name="is_training")
        
        self.global_step = tf.compat.v1.Variable(0, trainable=False, name="Global_Step")
        self.initializer=tf.compat.v1.keras.initializers.VarianceScaling()
        
        
        #题目初始化向量   shape  [number of questions,  dimentions]
        self.skill_w = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling()([840,256]),dtype=tf.float32, trainable=True, name = 'skill_w')
        zero_skill = tf.zeros((1, 256))  #[padding]

        all_skill = tf.concat([self.skill_w, zero_skill], axis = 0)

        skill = tf.nn.embedding_lookup(all_skill, self.input_id)
        next_skill = tf.nn.embedding_lookup(all_skill, self.valid_id)
        
        #遮罩向量
        zeros = tf.ones((2,1)) 
        ones = tf.ones((1,1)) *(-2**32+1)
        ttt = tf.concat([zeros,ones],axis = 0)
        masked = tf.nn.embedding_lookup(ttt, self.x_answer)
        masked = tf.reshape(masked, [-1, 29])

        x_answer1 = tf.tile(self.x_answer1, [1,1,10])
        input_data = tf.concat([skill ,x_answer1],axis = -1)
        input_data = tf.compat.v1.layers.dense(input_data, units = 256)
        input_data = tf.nn.relu(input_data) 
        input_data = tf.nn.dropout(input_data, 0.5)

        outputs = multi_span(input_data, self.dropout_keep_prob, is_training = self.is_training) 


        alpha = tf.matmul(skill,  tf.expand_dims(next_skill, axis = -1))
        alpha = tf.reshape(alpha, [-1, 29])
       # alpha = alpha*masked
        alpha = tf.nn.softmax(alpha)
        outputs = tf.matmul(tf.transpose(outputs, [0,2,1]), tf.expand_dims(alpha, axis = -1))
        outputs = tf.reshape(outputs, [-1, self.hidden_size])
        
            
        outputs = tf.compat.v1.layers.dense(outputs, units = 256)
        outputs = tf.nn.relu(outputs)
        
        print(np.shape(next_skill))
        outputs = tf.concat([next_skill, outputs],axis = -1)
        
        outputs = tf.compat.v1.layers.dense(outputs, units = 512)
        outputs = tf.nn.relu(outputs)   
        outputs = tf.compat.v1.layers.dense(outputs, units = 128)
        outputs = tf.nn.relu(outputs)   


        self.logits = tf.compat.v1.layers.dense(outputs, units = 2)
        print('aa')
        print(np.shape(self.logits))



        #make prediction
        self.pred = tf.sigmoid(self.logits, name="pred")

        # loss function
        #self.loss = tf.reduce_sum(tf.abs(self.logits - self.target_correctness))
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.target_correctness1), name="losses") 
        self.cost = self.loss
