import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from nn_layer import softmax_layer, bi_dynamic_rnn, reduce_mean_with_len
from att_layer import bilinear_attention_layer, dot_produce_attention_layer
from config import *
from utils import load_w2v, batch_index, load_inputs_twitter
import os, sys

sys.path.append(os.getcwd())

'''
Code is based on and originally written by Lisa Meijer (https://github.com/lhmeijer/ABSCEM) and Maria Mihaela Trusca (https://github.com/mtrusca/HAABSA_PLUS_PLUS).
Adapted by Kunal Geed
'''
class NeuralLanguageModel:
    glob_var_best_iter = 0

    def set_best_iter(self, iter):
        global glob_var_best_iter
        glob_var_best_iter = iter

    def get_global(self):
        return glob_var_best_iter

    def lcr_rot(self, input_fw, input_bw, sen_len_fw, sen_len_bw, target, sen_len_tr, keep_prob1, keep_prob2, l2,
                _id='all'):
        # prob, att_l, att_r, att_t_l, att_t_r, Layer_information = None
        # return prob, att_l, att_r, att_t_l, att_t_r, Layer_information
        return None, None, None, None, None, None

    '''
    Train the model using the training data and save the best model.
    
    '''

    def fit(self, train_path, test_path, accuracyOnt, test_size, remaining_size, useOntology=False, learning_rate=0.09,
            keep_prob=0.3, momentum=0.85, l2=0.00001):
        print_config()
        with tf.device('/cpu:0'):
            word_id_mapping, w2v = load_w2v(FLAGS.embedding_path,
                                            FLAGS.embedding_dim)  # WordEmbedding = word dictionary and w2v
            word_embedding = tf.constant(w2v, name='word_embedding')  # Defining tensorflow constant.
            #The keep probs
            keep_prob1 = tf.placeholder(tf.float32, name="keep_prob1")
            keep_prob2 = tf.placeholder(tf.float32, name="keep_prob2")

            with tf.name_scope('inputs'):
                #Defining the Input tensors
                x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len], name='x')
                y = tf.placeholder(tf.float32, [None, FLAGS.n_class], name='y')
                sen_len = tf.placeholder(tf.int32, [None], name='sen_len')
                #Right sentence and right sentence length
                x_bw = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len], name='x_bw')
                sen_len_bw = tf.placeholder(tf.int32, [None], name='sen_len_bw')
                #Target Words and Target Length
                target_words = tf.placeholder(tf.int32, [None, FLAGS.max_target_len], name='target_words')
                tar_len = tf.placeholder(tf.int32, [None], name='tar_len')
            # Looking up the embeddings.
            inputs_fw = tf.nn.embedding_lookup(word_embedding, x)  # number_of_words x bert_embeddings_dim matrix
            inputs_bw = tf.nn.embedding_lookup(word_embedding, x_bw)
            target = tf.nn.embedding_lookup(word_embedding, target_words)

            alpha_fw, alpha_bw = None, None
            #Calling the model itself
            prob, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r, layer_information = self.lcr_rot(inputs_fw, inputs_bw,
                                                                                             sen_len, sen_len_bw,
                                                                                             target,
                                                                                             tar_len, keep_prob1,
                                                                                             keep_prob2, l2, 'all')
            # saving procedure for the tensorflow graph
            model_name = "LCRv4"
            tf.add_to_collection(model_name + "_prob", prob)
            tf.add_to_collection(model_name + "_lembedding", layer_information['embedding_left'])
            tf.add_to_collection(model_name + "_rembedding", layer_information['embedding_right'])
            tf.add_to_collection(model_name + "_left_hidden", layer_information['left_hidden_state'])
            tf.add_to_collection(model_name + "_right_hidden", layer_information['right_hidden_state'])
            tf.add_to_collection(model_name + "_wsli", layer_information['weighted_states_left_initial'])
            tf.add_to_collection(model_name + "_wsri", layer_information['weighted_states_right_initial'])

            for i in range(2):
                tf.add_to_collection(model_name + "_wsl_" + str(i), layer_information["weighted_states_left_" + str(i)])
                tf.add_to_collection(model_name + "_wsr_" + str(i),
                                     layer_information["weighted_states_right_" + str(i)])
            #Loss Function and accuracy functions used to calculate the accuracy of the techniques.
            loss = loss_func(y, prob)
            acc_num, acc_prob = acc_func(y, prob)
            global_step = tf.Variable(0, name='tr_global_step', trainable=False)
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(loss,
                                                                                                            global_step=global_step)
            #The true y and and the predicted y. We take the argmax
            true_y = tf.argmax(y, 1)
            pred_y = tf.argmax(prob, 1)

            title = '-d1-{}d2-{}b-{}r-{}l2-{}sen-{}dim-{}h-{}c-{}'.format(
                FLAGS.keep_prob1,
                FLAGS.keep_prob2,
                FLAGS.batch_size,
                FLAGS.learning_rate,
                FLAGS.l2_reg,
                FLAGS.max_sentence_len,
                FLAGS.embedding_dim,
                FLAGS.n_hidden,
                FLAGS.n_class
            )

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        #Creating saver to save the Model to be used to predict later.
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            import time
            timestamp = str(int(time.time()))
            _dir = 'summary/' + str(timestamp) + '_' + title
            test_loss = tf.placeholder(tf.float32)
            test_acc = tf.placeholder(tf.float32)
            train_summary_op, test_summary_op, validate_summary_op, train_summary_writer, test_summary_writer, \
            validate_summary_writer = summary_func(loss, acc_prob, test_loss, test_acc, _dir, title, sess)

            save_dir = 'temp_model/' + str(timestamp) + '_' + title + '/'
            # saver = saver_func(save_dir)

            sess.run(tf.global_variables_initializer())
            # saver.restore(sess, '/-')

            if FLAGS.is_r == '1':
                is_r = True
            else:
                is_r = False

            tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len, _, _, _, _ = load_inputs_twitter(
                train_path,
                word_id_mapping,
                FLAGS.max_sentence_len,
                'TC',
                is_r,
                FLAGS.max_target_len
            )
            te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word, te_tar_len, _, _, _, _ = load_inputs_twitter(
                test_path,
                word_id_mapping,
                FLAGS.max_sentence_len,
                'TC',
                is_r,
                FLAGS.max_target_len
            )

            def get_batch_data(x_f, sen_len_f, x_b, sen_len_b, yi, target, tl, batch_size, kp1, kp2, is_shuffle=True):
                for index in batch_index(len(yi), batch_size, 1, is_shuffle):
                    feed_dict = {
                        x: x_f[index],
                        x_bw: x_b[index],
                        y: yi[index],
                        sen_len: sen_len_f[index],
                        sen_len_bw: sen_len_b[index],
                        target_words: target[index],
                        tar_len: tl[index],
                        keep_prob1: kp1,
                        keep_prob2: kp2,
                    }
                    yield feed_dict, len(index)

            max_acc = 0.
            max_fw, max_bw = None, None
            max_tl, max_tr = None, None
            max_ty, max_py = None, None
            max_prob = None
            step = None
            for i in range(FLAGS.n_iter):
                trainacc, traincnt = 0., 0
                for train, numtrain in get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word,
                                                      tr_tar_len,
                                                      FLAGS.batch_size, keep_prob, keep_prob):
                    # _, step = sess.run([optimizer, global_step], feed_dict=train)
                    _, step, summary, _trainacc = sess.run([optimizer, global_step, train_summary_op, acc_num],
                                                           feed_dict=train)
                    train_summary_writer.add_summary(summary, step)
                    # embed_update = tf.assign(word_embedding, tf.concat(0, [tf.zeros([1, FLAGS.embedding_dim]), word_embedding[1:]]))
                    # sess.run(embed_update)
                    trainacc += _trainacc  # saver.save(sess, save_dir, global_step=step)
                    traincnt += numtrain

                acc, cost, cnt = 0., 0., 0
                fw, bw, tl, tr, ty, py = [], [], [], [], [], []
                p = []
                for test, num in get_batch_data(te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y,
                                                te_target_word, te_tar_len, 2000, 1.0, 1.0, False):
                    if FLAGS.method == 'TD-ATT' or FLAGS.method == 'IAN':
                        _loss, _acc, _fw, _bw, _tl, _tr, _ty, _py, _p = sess.run(
                            [loss, acc_num, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r, true_y, pred_y, prob],
                            feed_dict=test)
                        fw += list(_fw)
                        bw += list(_bw)
                        tl += list(_tl)
                        tr += list(_tr)
                    else:
                        _loss, _acc, _ty, _py, _p, _fw, _bw, _tl, _tr = sess.run(
                            [loss, acc_num, true_y, pred_y, prob, alpha_fw, alpha_bw, alpha_t_l, alpha_t_r],
                            feed_dict=test)

                    ty = np.asarray(_ty)
                    py = np.asarray(_py)
                    p = np.asarray(_p)
                    fw = np.asarray(_fw)
                    bw = np.asarray(_bw)
                    tl = np.asarray(_tl)
                    tr = np.asarray(_tr)
                    acc += _acc
                    cost += _loss * num
                    cnt += num
                print('all samples={}, correct prediction={}'.format(cnt, acc))
                trainacc = trainacc / traincnt
                acc = acc / cnt
                if useOntology:
                    totalacc = ((acc * remaining_size) + (accuracyOnt * (test_size - remaining_size))) / test_size
                else:
                    totalacc = acc
                cost = cost / cnt
                print(
                    'Iter {}: mini-batch loss={:.6f}, train acc={:.6f}, test acc={:.6f}, combined acc={:.6f}'.format(i,
                                                                                                                     cost,
                                                                                                                     trainacc,
                                                                                                                     acc,
                                                                                                                     totalacc))
                summary = sess.run(test_summary_op, feed_dict={test_loss: cost, test_acc: acc})
                test_summary_writer.add_summary(summary, step)
                print("Accuracy: " + str(acc))
                if totalacc > max_acc:  # When the current accuracy is better
                    max_acc = totalacc
                    max_fw = fw  # Best embeddings you managed to get.
                    max_bw = bw
                    max_tl = tl
                    max_tr = tr
                    max_ty = ty
                    max_py = py
                    max_prob = p
                    self.set_best_iter(i)
                    file_to_save_model = "results/tf_model" ".ckpt-1000"
                    saver.save(sess, file_to_save_model)
            print('Optimization Finished! Max acc={}'.format(max_acc))

            print('Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
                FLAGS.learning_rate,
                FLAGS.n_iter,
                FLAGS.batch_size,
                FLAGS.n_hidden,
                FLAGS.l2_reg
            ))
            sess.close()

            return max_acc, np.where(np.subtract(max_py, max_ty) == 0, 0,
                                     1), max_fw.tolist(), max_bw.tolist(), max_tl.tolist(), max_tr.tolist()

    def predict(self, tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_target_word, tr_tar_len, kp1, kp2):
        print("predict")
        tf.reset_default_graph()
        model_name = "LCRv4"
        graph = tf.Graph()
        session = tf.Session(graph=graph)
        #best_iter = self.get_global()
        print('Best_Iteration was: ')
        file_to_save_model = "results/tf_model" + ".ckpt-1000"

        with graph.as_default():
            new_saver = tf.train.import_meta_graph(file_to_save_model + ".meta")
            #word_id_mapping, w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)
            #word_embedding = tf.constant(w2v, name='word_embedding')
            collect = list()
            collect.append(tf.get_collection(model_name + "_prob"))
            collect.append(tf.get_collection(model_name + "_lembedding"))
            collect.append(tf.get_collection(model_name + "_rembedding"))
            collect.append(tf.get_collection(model_name + "_left_hidden"))
            collect.append(tf.get_collection(model_name + "_right_hidden"))
            collect.append(tf.get_collection(model_name + "_wsli"))
            collect.append(tf.get_collection(model_name + "_wsri"))

            for i in range(2):
                collect.append(tf.get_collection(model_name + "_wsl_" + str(i)))
                collect.append(tf.get_collection(model_name + "_wsr_" + str(i)))

            new_saver.restore(session, file_to_save_model)

            keep_prob1 = session.graph.get_tensor_by_name("keep_prob1:0")
            keep_prob2 = session.graph.get_tensor_by_name("keep_prob2:0")
            x = session.graph.get_tensor_by_name("inputs/x:0")
            sen_len = session.graph.get_tensor_by_name("inputs/sen_len:0")
            x_bw = session.graph.get_tensor_by_name("inputs/x_bw:0")
            sen_len_bw = session.graph.get_tensor_by_name("inputs/sen_len_bw:0")
            target_words = session.graph.get_tensor_by_name("inputs/target_words:0")
            tar_len = session.graph.get_tensor_by_name("inputs/tar_len:0")
            #inputs_fw = tf.nn.embedding_lookup(word_embedding, x)
            #inputs_bw = tf.nn.embedding_lookup(word_embedding, x_bw)
            #target = tf.nn.embedding_lookup(word_embedding, target_words)
            feed_dict = {
                x: tr_x,
                x_bw: tr_x_bw,
                sen_len: tr_sen_len,
                sen_len_bw: tr_sen_len_bw,
                target_words: tr_target_word,
                tar_len: tr_tar_len,
                keep_prob1: kp1,
                keep_prob2: kp2
            }
            result_of_collections = session.run(collect, feed_dict=feed_dict)
            session.close()
            predictions = result_of_collections[0][0]
            # model_name+"_lembedding"
            layer_information = {
                'left_embedding': result_of_collections[1],
                'right_embedding': result_of_collections[2],
                'left_hidden_state': result_of_collections[3],
                'right_hidden_state': result_of_collections[4],
                'left_weighted_state_ini': result_of_collections[5],
                'right_weighted_state_ini': result_of_collections[6],
                'left_weighted_state_0': result_of_collections[7],
                'right_weighted_state_0': result_of_collections[8],
                'left_weighted_state_1': result_of_collections[9],
                'right_weighted_state_1': result_of_collections[10]
            }

        return predictions, layer_information
