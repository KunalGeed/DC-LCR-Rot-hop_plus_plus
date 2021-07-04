import pickle
from sklearn.metrics import f1_score
from config import *
from utils import *
import numpy as np

'''
Code is based on the work done by Lisa Meijer (https://github.com/lhmeijer/ABSCEM)
Adapted by Kunal Geed
'''

def generate_hypothesis(sentences_left, sentences_right, relation_left, relation_right):
    '''
    :param sentences_left: words in the left context
    :param sentences_right: words in the right context
    :param relation_left: relation hypothesis left
    :param relation_right: relation hypothesis right
    :return: The hypothesis
    '''
    # Five left hypothesis
    pos_tags_left = []
    word_sentiment_left = []
    mentions_left = []
    onttag = OntologyTags()
    for sen in sentences_left:
        pos_tags_left = pos_tag_sentence(sen, pos_tags_left)
        mentions_left = onttag.mention_tagging(sen, mentions_left)
        word_sentiment_left = onttag.sentence_word_sentiment(sen, word_sentiment_left)
    pos_tags_right = []
    word_sentiment_right = []
    mentions_right = []
    for sen in sentences_right:
        pos_tags_right = pos_tag_sentence(sen, pos_tags_right)
        mentions_right = onttag.mention_tagging(sen, mentions_right)
        word_sentiment_right = onttag.sentence_word_sentiment(sen, word_sentiment_right)

    aspect_senti_left, aspect_senti_right = make_SentiAspect_dataset(relation_left, relation_right, word_sentiment_left,
                                                                     word_sentiment_right)
    changed_hypo_left = []
    changed_hypo_right = []
    for t in relation_right:
        if t == 'no':
            changed_hypo_right.append([0, 1])
        else:
            changed_hypo_right.append([1, 0])
    for p in relation_left:
        if p == 'no':
            changed_hypo_left.append([0, 1])
        else:
            changed_hypo_left.append([1, 0])

    hypothesis = {'pos_tag_left': pos_tags_left, 'pos_tag_right': pos_tags_right,
                  'mentions_left': mentions_left, 'mentions_right': mentions_right,
                  'sentiment_left': word_sentiment_left, 'sentiment_right': word_sentiment_right,
                  'aspect_left': changed_hypo_left, 'aspect_right': changed_hypo_right,
                  'sentiment_relation_left': aspect_senti_left, 'sentiment_relation_right': aspect_senti_right
                  }
    return hypothesis

print('Loading Data')
##############TRAIN#####################
filename = 'Layer_information_test'  # messed up the name when pickling the file
infile = open(filename, 'rb')
layer_information_train = pickle.load(infile)
infile.close()

filename = 'Predictions_training'
infile = open(filename, 'rb')
pred_y = pickle.load(infile)
infile.close()

filename = 'True_y_Training'
infile = open(filename, 'rb')
true_y = pickle.load(infile)
infile.close()
####################Test Data#########################
filename = 'Layer_information_training'
infile = open(filename, 'rb')
layer_information_test = pickle.load(infile)
infile.close()
filename = 'Sentence_length_test'
infile = open(filename, 'rb')
sentence_length_test = pickle.load(infile)
infile.close()
############################################LCR-ROT-HOP Accuracy########################################################
filename = 'Predictions_test'
infile = open(filename, 'rb')
Predictions_test = pickle.load(infile)
infile.close()
sentence_length_test_left = sentence_length_test['te_sen_len']
filename = 'Test_y_Training'
infile = open(filename, 'rb')
te_y = pickle.load(infile)
infile.close()

argmax_pred = np.argmax(te_y, axis=1)
argmax_true = np.argmax(Predictions_test, axis=1)
correct_one = 0
correct_two = 0
correct_zero = 0
correct_score = 0
cnt = 0
for i in range(len(te_y)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
        if (argmax_true[i] == 0):
            correct_zero = correct_zero + 1
        elif argmax_true[i] == 1:
            correct_one = correct_one + 1
        else:
            correct_two = correct_two + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('Checking total for each class')
print(len(argmax_true == 0))
print(len(argmax_true == 1))
print(len(argmax_true == 2))
print('accuracy LCR_ROT_HOP')
print(hitrate)
print('class accuracies')
print('0')
print(correct_zero)
print('1')
print(correct_one)
print('2')
print(correct_two)
#####################Correct predictions##########################
correct_predictions_training = []
correct_one = 0
correct_two = 0
correct_zero = 0
argmax_pred_y = np.argmax(pred_y, axis=1)
argmax_true_y = np.argmax(true_y, axis=1)
for x in range(len(argmax_true_y)):
    if argmax_pred_y[x] == argmax_true_y[x]:
        correct_predictions_training.append(1)
        if argmax_pred_y[x] == 0:
            correct_zero += 1
        elif argmax_true_y[x] == 1:
            correct_one += 1
        else:
            correct_two += 1
    else:
        correct_predictions_training.append(0)
print('Checking total for each class')
print(len(argmax_true_y == 0))
print(len(argmax_true_y == 1))
print(len(argmax_true_y == 2))
print('class accuracies')
print('0')
print(correct_zero)
print('1')
print(correct_one)
print('2')
print(correct_two)
print('Overall accuracy')
print(np.sum(correct_predictions_training))
print(len(correct_predictions_training))
print('Loaded Data')

word_id_mapping, w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)

tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len, _, _, _, original_sentences_left_tr, original_sentences_right_tr, \
original_words_left_tr, original_words_right_tr, original_targets_tr, aspects_left_tr, aspects_right_tr = \
    load_inputs_twitter_sentences(
        FLAGS.train_path,
        word_id_mapping,
        FLAGS.max_sentence_len,
        'TC',
        True,
        FLAGS.max_target_len)

te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word, te_tar_len, _, _, _, original_sentences_left_te, original_sentences_right_te, original_words_left_te, \
original_words_right_te, original_targets_te, aspects_left_te, aspects_right_te = \
    load_inputs_twitter_sentences(
        FLAGS.test_path,
        word_id_mapping,
        FLAGS.max_sentence_len,
        'TC',
        True,
        FLAGS.max_target_len
    )

###################Generating Training Data ###########################
layer_training, words_left, words_right = combine_layer_information(layer_information_train, original_words_left_tr,
                                                                    original_words_right_tr, tr_sen_len, tr_sen_len_bw)

sentences_left_training, lengths_left_training = edit_sentences(original_sentences_left_tr)
sentences_right_training, lengths_right_training = edit_sentences(original_sentences_right_tr)

training_hypothesis = generate_hypothesis(sentences_left_training, sentences_right_training, aspects_left_tr,
                                          aspects_right_tr)

###############################Generating Test Data####################################################################
layer_test, words_left_te, words_right_te = combine_layer_information(layer_information_test, original_words_left_te,
                                                                      original_words_right_te, te_sen_len,
                                                                      te_sen_len_bw)

sentences_left_test, lengths_left_test = edit_sentences(original_sentences_left_te)
sentences_right_test, lengths_right_test = edit_sentences(original_sentences_right_te)

te_hypothesis = generate_hypothesis(sentences_left_test, sentences_right_test, aspects_left_te, aspects_right_te)
##################################################Statistics##########################################################
print('Words LEFT and RIGHT: Test')
print(len(te_hypothesis['pos_tag_left']))
print(len(te_hypothesis['pos_tag_right']))
print(len(words_left_te))
print(len(words_right_te))
print(np.sum(lengths_left_test))
print(np.sum(lengths_right_test))
print('Words LEFT and RIGHT: TRAIN')
print(len(training_hypothesis['pos_tag_left']))
print(len(training_hypothesis['pos_tag_right']))
print(len(words_left))
print(len(words_right))
print(np.sum(lengths_left_training))
print(np.sum(lengths_right_training))

########################POS-Tagging Hypothesis#########################################################################
print('Starting POS-Tagging')
accuracies = []
optimized_hitrate = []
hl_tr = training_hypothesis['pos_tag_left']
hr_tr = training_hypothesis['pos_tag_right']
hl_te = te_hypothesis['pos_tag_left']
hr_te = te_hypothesis['pos_tag_right']
##################Embeddings################################
print('layer 1')
str1 = 'left_embedding'
str2 = 'right_embedding'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)
print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)

##################Hidden_States################################
print('layer 2')
str1 = 'left_hidden_state'
str2 = 'right_hidden_state'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)
print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=1000).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)

##################Weighted Hidden_States Ini################################
print('layer 3')
str1 = 'left_weighted_state_ini'
str2 = 'right_weighted_state_ini'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)
print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)
##################Weighted Hidden_States 0################################
print('layer 4')
str1 = 'left_weighted_state_0'
str2 = 'right_weighted_state_0'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)
print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)
##################Weighted Hidden_States 1################################
print('layer 5')
str1 = 'left_weighted_state_1'
str2 = 'right_weighted_state_1'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)

print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)
print(accuracies)
print('optimized_hitrate')
print(optimized_hitrate)
accuracies = []
optimized_hitrate = []
########################Mention Hypothesis#########################################################################
print('Starting Mention-Tagging')
hl_tr = training_hypothesis['mentions_left']
hr_tr = training_hypothesis['mentions_right']
hl_te = te_hypothesis['mentions_left']
hr_te = te_hypothesis['mentions_right']
##################Embeddings################################
print('layer 1')
str1 = 'left_embedding'
str2 = 'right_embedding'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)
print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)

##################Hidden_States################################
print('layer 2')
str1 = 'left_hidden_state'
str2 = 'right_hidden_state'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)
print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)

##################Weighted Hidden_States Ini################################
print('layer 3')
str1 = 'left_weighted_state_ini'
str2 = 'right_weighted_state_ini'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)
print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)
##################Weighted Hidden_States 0################################
print('layer 4')
str1 = 'left_weighted_state_0'
str2 = 'right_weighted_state_0'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)
print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)
##################Weighted Hidden_States 1################################
print('layer 5')
str1 = 'left_weighted_state_1'
str2 = 'right_weighted_state_1'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)

print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)
print(accuracies)
print('optimized_hitrate')
print(optimized_hitrate)
accuracies = []
optimized_hitrate = []

########################Sentiment Hypothesis#########################################################################
print('Starting sentiment_left')
hl_tr = training_hypothesis['sentiment_left']
hr_tr = training_hypothesis['sentiment_right']
hl_te = te_hypothesis['sentiment_left']
hr_te = te_hypothesis['sentiment_right']
##################Embeddings################################
print('layer 1')
str1 = 'left_embedding'
str2 = 'right_embedding'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)
print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)

##################Hidden_States################################
print('layer 2')
str1 = 'left_hidden_state'
str2 = 'right_hidden_state'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)
print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)

##################Weighted Hidden_States Ini################################
print('layer 3')
str1 = 'left_weighted_state_ini'
str2 = 'right_weighted_state_ini'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)
print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)
##################Weighted Hidden_States 0################################
print('layer 4')
str1 = 'left_weighted_state_0'
str2 = 'right_weighted_state_0'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)
print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)
##################Weighted Hidden_States 1################################
print('layer 5')
str1 = 'left_weighted_state_1'
str2 = 'right_weighted_state_1'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)

print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)
print('accuracies')
print(accuracies)
print('Optimized_hitrate')
print(optimized_hitrate)
accuracies = []
optimized_hitrate = []

########################Aspect Hypothesis#########################################################################
print('Starting aspect')
hl_tr = training_hypothesis['aspect_left']
hr_tr = training_hypothesis['aspect_right']
hl_te = te_hypothesis['aspect_left']
hr_te = te_hypothesis['aspect_right']
##################Embeddings################################
print('layer 1')
str1 = 'left_embedding'
str2 = 'right_embedding'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)
print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)

##################Hidden_States################################
print('layer 2')
str1 = 'left_hidden_state'
str2 = 'right_hidden_state'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)
print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)

##################Weighted Hidden_States Ini################################
print('layer 3')
str1 = 'left_weighted_state_ini'
str2 = 'right_weighted_state_ini'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)
print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)
##################Weighted Hidden_States 0################################
print('layer 4')
str1 = 'left_weighted_state_0'
str2 = 'right_weighted_state_0'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)
print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)
##################Weighted Hidden_States 1################################
print('layer 5')
str1 = 'left_weighted_state_1'
str2 = 'right_weighted_state_1'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)

print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)
print(accuracies)
print('optimized_hitrate')
print(optimized_hitrate)
accuracies = []
optimized_hitrate = []
########################Aspect_Sentiment Hypothesis#########################################################################
print('Starting aspect sentiment')
hl_tr = training_hypothesis['sentiment_relation_left']
hr_tr = training_hypothesis['sentiment_relation_right']
hl_te = te_hypothesis['sentiment_relation_left']
hr_te = te_hypothesis['sentiment_relation_right']
##################Embeddings################################
print('layer 1')
str1 = 'left_embedding'
str2 = 'right_embedding'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)
print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)

##################Hidden_States################################
print('layer 2')
str1 = 'left_hidden_state'
str2 = 'right_hidden_state'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)
print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)

##################Weighted Hidden_States Ini################################
print('layer 3')
str1 = 'left_weighted_state_ini'
str2 = 'right_weighted_state_ini'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)
print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)
##################Weighted Hidden_States 0################################
print('layer 4')
str1 = 'left_weighted_state_0'
str2 = 'right_weighted_state_0'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)
print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)
##################Weighted Hidden_States 1################################
print('layer 5')
str1 = 'left_weighted_state_1'
str2 = 'right_weighted_state_1'

x_corr_new, x_incorr, y_corr_new, y_incorr = make_dataset_edited(layer_training, hl_tr,
                                                                 hr_tr,
                                                                 lengths_left_training,
                                                                 lengths_right_training,
                                                                 correct_predictions_training
                                                                 , str1, str2)

x_test_new, y_test_new = make_dataset_test_edited(layer_test, hl_te, hr_te,
                                                  lengths_left_test,
                                                  lengths_right_test, str1, str2)

print(x_corr_new.shape)
#new_rate=hyperparameter_optimization(x_corr_new,y_corr_new,x_test_new,y_test_new)
#optimized_hitrate.append(new_rate)

clf = MLPClassifier(hidden_layer_sizes=(900,), random_state=1, max_iter=400).fit(x_corr_new, y_corr_new)
py = clf.predict(x_test_new)

argmax_pred = np.argmax(py, axis=1)
argmax_true = np.argmax(y_test_new, axis=1)
correct_score = 0
cnt = 0
for i in range(len(py)):
    if argmax_true[i] == argmax_pred[i]:
        correct_score = correct_score + 1
    cnt = cnt + 1

hitrate = correct_score / cnt
print('accuracy')
print(hitrate)
print('f1')
print(f1_score(y_test_new, py, average='weighted'))
accuracies.append(hitrate)
print(accuracies)
print('optimized_hitrate')
print(optimized_hitrate)