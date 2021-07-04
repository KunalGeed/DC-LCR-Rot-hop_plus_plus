from config import *
from lcr_v4 import LCRRotHopModel
from utils import *
import pickle

'''
This is the main file to generate the files to be used as input for diagnostic classifiers.

Code is based on the work done by Lisa Meijer (https://github.com/lhmeijer/ABSCEM) and Maria Mihaela Trusca (https://github.com/mtrusca/HAABSA_PLUS_PLUS).
It was adapted by me for this project
'''
test = FLAGS.test_path
train = FLAGS.train_path
accuracyOnt = 0.87
remaining_size = 250
test_size = 650
useOntology = False
lcr_v4_model = LCRRotHopModel()

accuracy_final, x, max_fw, max_bw, max_tl, layer_information = lcr_v4_model.fit(FLAGS.train_path, test, accuracyOnt,
                                                                           test_size,
                                                                          remaining_size, useOntology)
print('Max Accuracy(Final): ', accuracy_final)

word_id_mapping, w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)
tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len, _, _, _, _ = load_inputs_twitter(
    FLAGS.train_path,
    word_id_mapping,
    FLAGS.max_sentence_len,
    'TC',
    True,
    FLAGS.max_target_len)

te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word, te_tar_len, _, _, _, _ = load_inputs_twitter(
    FLAGS.test_path,
    word_id_mapping,
    FLAGS.max_sentence_len,
    'TC',
    True,
    FLAGS.max_target_len
)


#Keep probaility should be 1 when predicting as its only nessesary for drop-out and we do not want to do any regularization
kp1 = 1.0
kp2 = 1.0
print('Predicting')
prediction1, layer_information_tr = lcr_v4_model.predict(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw,
                                                       tr_target_word, tr_tar_len, kp1, kp2)


predictions2, layer_information_te = lcr_v4_model.predict(te_x, te_sen_len, te_x_bw, te_sen_len_bw,
                                                          te_target_word, te_tar_len, kp1, kp2)
print('Checking test accuracy')
argmax_pred = np.argmax(te_y, axis=1)
argmax_true = np.argmax(predictions2, axis=1)
correct_score = 0
cnt = 0
for i in range(len(te_y)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1
print(cnt)
print(correct_score)
hitrate = correct_score / cnt
print('accuracy')
print(hitrate)

#Comment remove the code below from string when you want to save the files.
'''
#Saving files to be used later. Can be used to generate our own pickle files or else use the files I provided.
print('Saving pickle files')

filename = "Layer_information_training"
outfile = open(filename, 'wb')
pickle.dump(layer_information_te, outfile)
outfile.close()
filename = "Layer_information_test"
outfile = open(filename, 'wb')
pickle.dump(layer_information_tr, outfile)
outfile.close()
filename = "Predictions_training"
outfile = open(filename, 'wb')
pickle.dump(prediction1, outfile)
outfile.close()
filename = "Predictions_test"
outfile = open(filename, 'wb')
pickle.dump(predictions2, outfile)
outfile.close()
filename = "True_y_Training"
outfile = open(filename, 'wb')
pickle.dump(tr_y, outfile)
outfile.close()
filename = "Test_y_Training"
outfile = open(filename, 'wb')
pickle.dump(te_y, outfile)
outfile.close()

sentence_length_train = {'tr_sen_len': tr_sen_len,
                         'tr_sen_len_bw': tr_sen_len_bw
                         }
sentence_length_test = {'te_sen_len': te_sen_len,
                        'te_sen_len_bw': te_sen_len_bw}

filename = "Sentence_length_train"
outfile = open(filename, 'wb')
pickle.dump(sentence_length_train, outfile)
outfile.close()
filename = "Sentence_length_test"
outfile = open(filename, 'wb')
pickle.dump(sentence_length_test, outfile)
outfile.close()


'''
