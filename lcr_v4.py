# !/usr/bin/env python
from NeuralLanguageModel import NeuralLanguageModel
# encoding: utf-8

import os, sys

sys.path.append(os.getcwd())

from nn_layer import softmax_layer, bi_dynamic_rnn, reduce_mean_with_len
from att_layer import bilinear_attention_layer, dot_produce_attention_layer
from config import *
from utils import load_w2v, batch_index, load_inputs_twitter
import numpy as np

tf.set_random_seed(1)

'''
Code Code is based on and originally written by Maria Mihaela Trusca (https://github.com/mtrusca/HAABSA_PLUS_PLUS).
Adapted by Kunal Geed
'''
class LCRRotHopModel(NeuralLanguageModel):

    def lcr_rot(self, input_fw, input_bw, sen_len_fw, sen_len_bw, target, sen_len_tr, keep_prob1, keep_prob2, l2,
                _id='all'):
        print('I am lcr_rot_altv4_inherited.')
        rate=1-keep_prob1
        cell = tf.contrib.rnn.LSTMCell
        input_fw_orignial = input_fw
        input_bw_original = input_bw
        # left hidden
        input_fw = tf.nn.dropout(input_fw, rate=rate)
        hiddens_l = bi_dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'l' + _id,
                                   'all')  # Hidden State Left, size = number of left words
        pool_l = reduce_mean_with_len(hiddens_l, sen_len_fw)

        # right hidden
        input_bw = tf.nn.dropout(input_bw, rate=rate)
        hiddens_r = bi_dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'r' + _id,
                                   'all')  # Hideen State Right (H_r) size=number of right words
        pool_r = reduce_mean_with_len(hiddens_r, sen_len_bw)

        # target hidden
        target = tf.nn.dropout(target, rate=rate)
        hiddens_t = bi_dynamic_rnn(cell, target, FLAGS.n_hidden, sen_len_tr, FLAGS.max_sentence_len, 't' + _id,
                                   'all')  # Hidden State Target (H_t) size= number of target words
        pool_t = reduce_mean_with_len(hiddens_t, sen_len_tr)

        # attention left
        att_l = bilinear_attention_layer(hiddens_l, pool_t, sen_len_fw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                         'tl')  # Attention Score left, size = 1 x number of left words
        weighted_hidden_states_left = tf.multiply(tf.transpose(att_l, perm=[0, 2, 1]),
                                                  hiddens_l)  # multiplied by the first attention score

        outputs_t_l_init = tf.matmul(att_l, hiddens_l)
        outputs_t_l = tf.squeeze(outputs_t_l_init)
        # attention right
        att_r = bilinear_attention_layer(hiddens_r, pool_t, sen_len_bw, 2 * FLAGS.n_hidden, l2, FLAGS.random_base,
                                         'tr')  # Attention Score right size= 1 x number of right words
        weighted_hidden_states_right = tf.multiply(tf.transpose(att_r, perm=[0, 2, 1]),
                                                   hiddens_r)  # multiplied by the first attention score at word level (not summed)

        outputs_t_r_init = tf.matmul(att_r, hiddens_r)
        outputs_t_r = tf.squeeze(outputs_t_r_init)

        # attention target left
        att_t_l = bilinear_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * FLAGS.n_hidden, l2,
                                           FLAGS.random_base,
                                           'l')  # Attention Score target left, size= 1 x number of target words.
        outputs_l_init = tf.matmul(att_t_l, hiddens_t)
        outputs_l = tf.squeeze(outputs_l_init)
        # attention target right
        att_t_r = bilinear_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * FLAGS.n_hidden, l2,
                                           FLAGS.random_base,
                                           'r')  # Attention Score target right size= 1 x number of target words.
        outputs_r_init = tf.matmul(att_t_r, hiddens_t)  # The hidden state times the attention score
        outputs_r = tf.squeeze(outputs_r_init)

        outputs_init_context = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
        outputs_init_target = tf.concat([outputs_l_init, outputs_r_init], 1)
        att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, l2,
                                                          FLAGS.random_base, 'fin1')  # alpha context
        att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, l2,
                                                         FLAGS.random_base, 'fin2')  # alpha target
        outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 0], 2), outputs_l_init))
        outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 1], 2), outputs_r_init))
        outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 0], 2), outputs_t_l_init))
        outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 1], 2), outputs_t_r_init))

        hierarchical_weighted_states_left = tf.multiply(tf.expand_dims(att_outputs_context[:, :, 0], 2),
                                                        weighted_hidden_states_left)
        hierarchical_weighted_states_right = tf.multiply(tf.expand_dims(att_outputs_context[:, :, 1], 2),
                                                         weighted_hidden_states_right)
        layer_information = {
            'embedding_left': input_fw_orignial,
            'embedding_right': input_bw_original,
            'left_hidden_state': hiddens_l,
            'right_hidden_state': hiddens_r,
            'weighted_states_left_initial': hierarchical_weighted_states_left,
            'weighted_states_right_initial': hierarchical_weighted_states_right
        }

        for i in range(2):
            # attention target
            att_l = bilinear_attention_layer(hiddens_l, outputs_l, sen_len_fw, 2 * FLAGS.n_hidden, l2,
                                             FLAGS.random_base, 'tl' + str(i))
            outputs_t_l_init = tf.matmul(att_l, hiddens_l)
            outputs_t_l = tf.squeeze(outputs_t_l_init)

            att_r = bilinear_attention_layer(hiddens_r, outputs_r, sen_len_bw, 2 * FLAGS.n_hidden, l2,
                                             FLAGS.random_base, 'tr' + str(i))
            outputs_t_r_init = tf.matmul(att_r, hiddens_r)
            outputs_t_r = tf.squeeze(outputs_t_r_init)

            # attention left
            att_t_l = bilinear_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * FLAGS.n_hidden, l2,
                                               FLAGS.random_base, 'l' + str(i))
            outputs_l_init = tf.matmul(att_t_l, hiddens_t)
            outputs_l = tf.squeeze(outputs_l_init)
            weighted_hidden_states_left = tf.multiply(tf.transpose(att_l, perm=[0, 2, 1]),
                                                      hiddens_l)  # multiplied by the first attention score

            # attention right
            att_t_r = bilinear_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * FLAGS.n_hidden, l2,
                                               FLAGS.random_base, 'r' + str(i))
            outputs_r_init = tf.matmul(att_t_r, hiddens_t)
            outputs_r = tf.squeeze(outputs_r_init)
            weighted_hidden_states_right = tf.multiply(tf.transpose(att_r, perm=[0, 2, 1]),
                                                       hiddens_r)  # multiplied by the first attention score at word level (not summed)

            outputs_init_context = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
            outputs_init_target = tf.concat([outputs_l_init, outputs_r_init], 1)
            att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, l2,
                                                              FLAGS.random_base, 'fin1' + str(i))  # alpha for target
            att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, l2,
                                                             FLAGS.random_base, 'fin2' + str(i))  # alpha for context
            outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 0], 2), outputs_l_init))
            outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:, :, 1], 2), outputs_r_init))
            outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 0], 2), outputs_t_l_init))
            outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:, :, 1], 2), outputs_t_r_init))
            hierarchical_weighted_states_left = tf.multiply(tf.expand_dims(att_outputs_context[:, :, 0], 2),
                                                            weighted_hidden_states_left)
            hierarchical_weighted_states_right = tf.multiply(tf.expand_dims(att_outputs_context[:, :, 1], 2),
                                                             weighted_hidden_states_right)
            layer_information['weighted_states_left_' + str(i)] = hierarchical_weighted_states_left
            layer_information['weighted_states_right_' + str(i)] = hierarchical_weighted_states_right

        outputs_fin = tf.concat([outputs_l, outputs_r, outputs_t_l, outputs_t_r], 1)
        prob = softmax_layer(outputs_fin, 8 * FLAGS.n_hidden, FLAGS.random_base, keep_prob2, l2, FLAGS.n_class)
        return prob, att_l, att_r, att_t_l, att_t_r, layer_information
