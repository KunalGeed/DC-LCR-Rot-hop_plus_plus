#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import nltk
from nltk import CoreNLPDependencyParser
from Ontology_Tags import OntologyTags
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

'''
Code is based on the work done by Lisa Meijer (https://github.com/lhmeijer/ABSCEM) and Maria Mihaela Trusca (https://github.com/mtrusca/HAABSA_PLUS_PLUS).
It was adapted by me for this project. I have added a few methods to the file.
'''
def batch_index(length, batch_size, n_iter=100, is_shuffle=True):
    index = list(range(length))
    for j in range(n_iter):
        if is_shuffle:
            np.random.shuffle(index)
        for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
            yield index[i * batch_size:(i + 1) * batch_size]


def load_word_id_mapping(word_id_file, encoding='utf8'):
    """
    :param word_id_file: word-id mapping file path
    :param encoding: file's encoding, for changing to unicode
    :return: word-id mapping, like hello=5
    """
    word_to_id = dict()
    for line in open(word_id_file):
        line = line.decode(encoding, 'ignore').lower().split()
        word_to_id[line[0]] = int(line[1])
    print('\nload word-id mapping done!\n')
    return word_to_id


def load_w2v(w2v_file, embedding_dim, is_skip=False):
    fp = open(w2v_file)
    if is_skip:
        fp.readline()
    w2v = []
    word_dict = dict()
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim)
    cnt = 0
    for line in fp:  # For every word in the word embedding file, we go through it.
        cnt += 1
        line = line.split()  # We split it for each line (into a vector)
        # line = line.split()
        if len(line) != embedding_dim + 1:
            print('a bad word embedding: {}'.format(line[0]))
            continue
        w2v.append([float(v) for v in line[1:]])  # cause at 0 we have the word. After that we add the rest
        word_dict[line[0]] = cnt  # not count but index.
    w2v = np.asarray(w2v, dtype=np.float32)
    w2v = np.row_stack((w2v, np.sum(w2v, axis=0) / cnt))
    print(np.shape(w2v))
    word_dict['$t$'] = (cnt + 1)
    # w2v -= np.mean(w2v, axis=0)
    # w2v /= np.std(w2v, axis=0)
    print(word_dict['$t$'], len(w2v))
    return word_dict, w2v


def load_word_embedding(word_id_file, w2v_file, embedding_dim, is_skip=False):
    word_to_id = load_word_id_mapping(word_id_file)
    word_dict, w2v = load_w2v(w2v_file, embedding_dim, is_skip)
    cnt = len(w2v)
    for k in word_to_id.keys():
        if k not in word_dict:
            word_dict[k] = cnt
            w2v = np.row_stack((w2v, np.random.uniform(-0.01, 0.01, (embedding_dim,))))
            cnt += 1
    print(len(word_dict), len(w2v))
    return word_dict, w2v


def change_y_to_onehot(y, pos_neu_neg=True):
    from collections import Counter
    print(Counter(y))
    if (pos_neu_neg):
        class_set = {'-1', '0', '1'}
    else:
        class_set = set(y)
    n_class = len(class_set)
    y_onehot_mapping = dict(zip(class_set, range(n_class)))
    print(y_onehot_mapping)
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)


def load_inputs_twitter(input_file, word_id_file, sentence_len, type_='', is_r=True, target_len=10, encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print('load word-to-id done!')
    # sum=0
    x, y, sen_len = [], [], []
    x_r, sen_len_r = [], []
    target_words = []
    tar_len = []
    all_target, all_sent, all_y = [], [], []
    pos_left_right = []  # First we append all the left words, then all the right words for the sentence. This is done for the whole dataset
    # read in txt file
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 3):
        # targets
        words = lines[i + 1].lower().split()
        target = words

        target_word = []
        for w in words:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        # sentiment
        y.append(lines[i + 2].strip().split()[0])

        # left and right context
        words = lines[i].lower().split()
        sent = words
        words_l, words_r = [], []
        flag = True
        for word in words:
            if word == '$t$':
                flag = False
                continue
            if flag:
                if word in word_to_id:
                    words_l.append(word_to_id[word])
                    # pos_left_right.append(pos_tagging(word))
            else:
                if word in word_to_id:
                    words_r.append(word_to_id[word])
                    # pos_left_right.append(pos_tagging(word))

        if type_ == 'TD' or type_ == 'TC':
            # words_l.extend(target_word)
            words_l = words_l[:sentence_len]
            words_r = words_r[:sentence_len]
            sen_len.append(len(words_l))
            # sum=sum+len(words_l)
            x.append(words_l + [0] * (sentence_len - len(words_l)))
            # tmp = target_word + words_r
            tmp = words_r
            # sum=sum+len(tmp)
            if is_r:
                tmp.reverse()
            sen_len_r.append(len(tmp))
            x_r.append(tmp + [0] * (sentence_len - len(tmp)))
            all_sent.append(sent)
            all_target.append(target)
        else:
            words = words_l + target_word + words_r
            words = words[:sentence_len]
            sen_len.append(len(words))
            x.append(words + [0] * (sentence_len - len(words)))
    all_y = y
    y = change_y_to_onehot(y)
    # print(sum)
    if type_ == 'TD':
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), \
               np.asarray(sen_len_r), np.asarray(y)
    elif type_ == 'TC':
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), np.asarray(sen_len_r), \
               np.asarray(y), np.asarray(target_words), np.asarray(tar_len), np.asarray(all_sent), np.asarray(
            all_target), np.asarray(all_y), np.asarray(pos_left_right)
    elif type_ == 'IAN':
        return np.asarray(x), np.asarray(sen_len), np.asarray(target_words), \
               np.asarray(tar_len), np.asarray(y)
    else:
        return np.asarray(x), np.asarray(sen_len), np.asarray(y)


def pos_tag_sentence(sentence, pos_tag_list, remove_numbering=True):
    '''
    Code is based on the work done by Lisa Meijer (https://github.com/lhmeijer/ABSCEM)
    Adapted by Kunal Geed
    :param sentence: Sentence
    :param pos_tag_list: list to append the part of speech tags to
    :param remove_numbering:  remove bert numbering
    :return: the pos_tag list updated with the words in the current sentence.
    '''
    if remove_numbering:
        sentence = remove_bert_numbering(sentence)
    tagged_sentence = nltk.pos_tag(sentence)
    for word_tag in tagged_sentence:
        if (word_tag[1].startswith('V')):  # Verb
            pos_tag_list.append([1, 0, 0, 0, 0])
        elif (word_tag[1].startswith('J')): #Adjective
            pos_tag_list.append([0, 1, 0, 0, 0])
        elif (word_tag[1].startswith('R')): #Adverb
            pos_tag_list.append([0, 0, 1, 0, 0])
        elif (word_tag[1].startswith('N')): #Noun
            pos_tag_list.append([0, 0, 0, 1, 0])
        else:#Remaining
            pos_tag_list.append([0, 0, 0, 0, 1])
    return pos_tag_list


def remove_bert_numbering(sentence):
    '''
    :param sentence: Sentence (tokenized)
    :return: remove the bert numbering from the tokens
    '''
    sentence_edited = []
    for sen in sentence:
        sen = sen.split('_', 1)[0]
        sentence_edited.append(sen)
    return sentence_edited


def load_inputs_twitter_v1(input_file, word_id_file, sentence_len, type_='', is_r=True, target_len=10,
                              get_sentence=False,
                              encoding='utf8'):
    '''
    Code is based on the work done by Maria Mihaela Trusca (https://github.com/mtrusca/HAABSA_PLUS_PLUS) and adapted by Kunal Geed
    :param input_file:
    :param word_id_file:
    :param sentence_len:
    :param type_:
    :param is_r:
    :param target_len:
    :param get_sentence:
    :param encoding:
    :return:
    '''
    print('starting kunal')
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print('load word-to-id done!')
    # sum=0
    onttag = OntologyTags()
    x, y, sen_len = [], [], []
    x_r, sen_len_r = [], []
    target_words = []
    tar_len = []
    all_target, all_sent, all_y = [], [], []
    pos_left_right = []  # First we append all the left words, then all the right words for the sentence. This is done for the whole dataset
    # read in txt file
    lines = open(input_file).readlines()
    all_sentences = []
    pos_tag_left = []
    pos_tag_right = []
    mentions_left = []
    mentions_right = []
    sentiment_word_right = []
    sentiment_word_left = []
    aspect_relation_left = []
    aspect_relation_right = []
    Original_data_left = []
    Original_data_right = []
    for i in range(0, len(lines), 3):
        # targets
        words = lines[i + 1].lower().split()
        target = words
        target_word_ori = []
        target_word = []
        for w in words:
            if w in word_to_id:
                target_word.append(word_to_id[w])
                target_word_ori.append(w)
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        # sentiment
        y.append(lines[i + 2].strip().split()[0])

        # left and right context
        words = lines[i].lower().split()
        sent = words
        words_l, words_r = [], []
        words_l_ori, words_r_ori = [], []
        flag = True
        all_sentences.append(words)
        for word in words:
            if word == '$t$':
                flag = False
                continue
            if flag:
                if word in word_to_id:
                    words_l.append(word_to_id[word])
                    words_l_ori.append(word)
            else:
                if word in word_to_id:
                    words_r.append(word_to_id[word])
                    words_r_ori.append(word)
        # Words in left part of the sentence
        Original_data_left.append(words_l_ori)  # Appeneding Left sentence
        Original_data_right.append(words_r_ori)  # Appending right sentence
        concatenated_sentence, aspect_indices = sentence_concatenate(words_l_ori, target_word_ori, words_r_ori)
        relation = lemmatize_and_pos_tagging(concatenated_sentence, aspect_indices)
        cnt = 0
        start_index = aspect_indices[0][0]
        if (len(aspect_indices[0]) > 1):
            end_index = aspect_indices[0][-1]
        else:
            end_index = start_index
        for i in range(len(relation[0])):
            if i < start_index:
                aspect_relation_left.append(relation[0][i])
            elif i > end_index:
                aspect_relation_right.append(relation[0][i])
            else:
                continue

        pos_tag_left = pos_tag_sentence(words_l_ori, pos_tag_left)
        pos_tag_right = pos_tag_sentence(words_r_ori, pos_tag_right)
        mentions_left = onttag.mention_tagging(words_l_ori, mentions_left)
        mentions_right = onttag.mention_tagging(words_r_ori, mentions_right)
        sentiment_word_left = onttag.sentence_word_sentiment(words_l_ori, sentiment_word_left)
        sentiment_word_right = onttag.sentence_word_sentiment(words_r_ori, sentiment_word_right)

        if type_ == 'TD' or type_ == 'TC':
            # words_l.extend(target_word)
            words_l = words_l[:sentence_len]
            words_r = words_r[:sentence_len]
            sen_len.append(len(words_l))
            # sum=sum+len(words_l)
            x.append(words_l + [0] * (sentence_len - len(words_l)))
            # tmp = target_word + words_r
            tmp = words_r
            # sum=sum+len(tmp)
            if is_r:
                tmp.reverse()
            sen_len_r.append(len(tmp))
            x_r.append(tmp + [0] * (sentence_len - len(tmp)))
            all_sent.append(sent)
            all_target.append(target)
        else:
            words = words_l + target_word + words_r
            words = words[:sentence_len]
            sen_len.append(len(words))
            x.append(words + [0] * (sentence_len - len(words)))
    all_y = y
    y = change_y_to_onehot(y)
    # print(sum)
    if type_ == 'TD':
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), \
               np.asarray(sen_len_r), np.asarray(y)
    elif type_ == 'TC':
        return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), np.asarray(sen_len_r), \
               np.asarray(y), np.asarray(target_words), np.asarray(tar_len), np.asarray(all_sent), np.asarray(
            all_target), np.asarray(all_y), np.asarray(all_sentences), np.asarray(pos_tag_right), np.asarray(
            pos_tag_left), \
               np.asarray(mentions_left), np.asarray(mentions_right), np.asarray(sentiment_word_left), np.asarray(
            sentiment_word_right), np.asarray(aspect_relation_left), np.asarray(aspect_relation_right)
    elif type_ == 'IAN':
        return np.asarray(x), np.asarray(sen_len), np.asarray(target_words), \
               np.asarray(tar_len), np.asarray(y)
    else:
        return np.asarray(x), np.asarray(sen_len), np.asarray(y)


def load_inputs_twitter_sentences(input_file, word_id_file, sentence_len, type_='', is_r=True, target_len=10,
                                  encoding='utf8'):
    '''
    Code is based on the work done by Maria Mihaela Trusca (https://github.com/mtrusca/HAABSA_PLUS_PLUS) and adapted by Kunal Geed.
    This is the final version used in the code.
    :param input_file:  Input files
    :param word_id_file:  Word ID files
    :param sentence_len: Sentence lengths
    :param type_: TC
    :param is_r: True (boolean value)
    :param target_len: target length
    :param encoding:
    :return: the inputs pre-processed and divided
    '''
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    print('load word-to-id done!')
    # sum=0
    x, y, sen_len = [], [], []
    x_r, sen_len_r = [], []
    target_words = []
    tar_len = []
    all_target, all_sent, all_y = [], [], []
    Original_data_left = []
    Original_data_right = []
    Original_words_right = []
    Original_words_left = []
    Original_targets = []
    aspect_relation_left = []
    aspect_relation_right = []
    # read in txt file
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 3): #All sentences
        # targets
        words = lines[i + 1].lower().split()
        Original_targets.append(words)
        target = words
        target_word = []
        target_word_ori = []
        #Saving Targets
        for w in words:
            if w in word_to_id:
                target_word.append(word_to_id[w])
                target_word_ori.append(w)
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        # sentiment
        y.append(lines[i + 2].strip().split()[0])

        # left and right context
        words = lines[i].lower().split()
        sent = words
        words_l, words_r = [], []
        words_l_ori, words_r_ori = [], []

        flag = True
        for word in words:
            if word == '$t$':
                flag = False
                continue
            if flag:
                if word in word_to_id:
                    words_l.append(word_to_id[word])
                    words_l_ori.append(word)
                    Original_words_left.append(word)
            else:
                if word in word_to_id:
                    words_r.append(word_to_id[word])
                    words_r_ori.append(word)
                    Original_words_right.append((word))
        #Saving the data for hypothesis
        Original_data_left.append(words_l_ori)  # Appeneding Left sentence
        Original_data_right.append(words_r_ori)  # Appending right sentence
        edited_words_l_ori, _ = combine_word_pieces_sentence(words_l_ori)
        edited_words_r_ori, _ = combine_word_pieces_sentence(words_r_ori)
        edited_target_ori, _ = combine_word_pieces_sentence(target_word_ori)
        concatenated_sentence, aspect_indices = sentence_concatenate(edited_words_l_ori, edited_target_ori,
                                                                     edited_words_r_ori)
        relation = lemmatize_and_pos_tagging(concatenated_sentence, aspect_indices)
        start_index = aspect_indices[0][0]
        if (len(aspect_indices[0]) > 1):
            end_index = aspect_indices[0][-1]
        else:
            end_index = start_index
        for i in range(len(relation[0])):
            if i < start_index:
                aspect_relation_left.append(relation[0][i])
            elif i > end_index:
                aspect_relation_right.append(relation[0][i])
            else:
                continue

        if type_ == 'TD' or type_ == 'TC':
            # words_l.extend(target_word)
            words_l = words_l[:sentence_len]
            words_r = words_r[:sentence_len]
            sen_len.append(len(words_l))
            # sum=sum+len(words_l)
            x.append(words_l + [0] * (sentence_len - len(words_l)))
            # tmp = target_word + words_r
            tmp = words_r
            # sum=sum+len(tmp)
            if is_r:
                tmp.reverse()
            sen_len_r.append(len(tmp))
            x_r.append(tmp + [0] * (sentence_len - len(tmp)))
            all_sent.append(sent)
            all_target.append(target)
        else:
            words = words_l + target_word + words_r
            words = words[:sentence_len]
            sen_len.append(len(words))
            x.append(words + [0] * (sentence_len - len(words)))
    all_y = y
    y = change_y_to_onehot(y)
    # print(sum)

    return np.asarray(x), np.asarray(sen_len), np.asarray(x_r), np.asarray(sen_len_r), \
           np.asarray(y), np.asarray(target_words), np.asarray(tar_len), np.asarray(all_sent), np.asarray(
        all_target), np.asarray(all_y), np.asarray(Original_data_left), np.asarray(Original_data_right), np.asarray(
        Original_words_left), \
           np.asarray(Original_words_right), np.asarray(Original_targets), np.asarray(aspect_relation_left), np.asarray(
        aspect_relation_right)


def lemmatize_and_pos_tagging(sentence, aspect_indices):
    '''
    Code is based on the work done by Lisa Meijer (https://github.com/lhmeijer/ABSCEM) and adapted by Kunal Geed
    :param sentence: Complete Sentence (tokenized)
    :param aspect_indices: target indices
    :return: Relation Hypothesis
    '''
    sentence = process_sentence_dep(sentence)
    core_nlp_dependency_parser = CoreNLPDependencyParser('http://localhost:9000')
    parses = core_nlp_dependency_parser.parse(sentence)
    dependencies = [[(governor, dep, dependent) for governor, dep, dependent in parse.triples()]
                    for parse in parses][0]

    aspects_dependencies = [['no'] * len(sentence) for i in range(len(aspect_indices))]

    backup_sentence = sentence.copy()
    interesting_translates = {'-LRB-': '(', '-RRB-': ')', '2\xa01/2': '2 1/2', "''": '"', ':-RRB-': ':)'}

    for dependency in dependencies:

        words = [dependency[0][0], dependency[2][0]]

        if words[0] in interesting_translates:
            words[0] = interesting_translates[words[0]]
        if words[1] in interesting_translates:
            words[1] = interesting_translates[words[1]]

        if words[0] in sentence:
            index_of_word1 = sentence.index(words[0])
            sentence[index_of_word1] = ''
        else:
            index_of_word1 = backup_sentence.index(words[0])

        if words[1] in sentence:
            index_of_word2 = sentence.index(words[1])
            sentence[index_of_word2] = ''
        else:
            index_of_word2 = backup_sentence.index(words[1])

        for aspect_index in range(len(aspect_indices)):

            if index_of_word1 in aspect_indices[aspect_index] and index_of_word2 not in \
                    aspect_indices[aspect_index]:
                aspects_dependencies[aspect_index][index_of_word2] = dependency[1]
            elif index_of_word1 not in aspect_indices[aspect_index] and index_of_word2 in \
                    aspect_indices[aspect_index]:
                aspects_dependencies[aspect_index][index_of_word1] = dependency[1]
            elif index_of_word1 in aspect_indices[aspect_index] and index_of_word2 in aspect_indices[aspect_index]:
                if aspects_dependencies[aspect_index][index_of_word1] == 'no':
                    aspects_dependencies[aspect_index][index_of_word1] = dependency[1]
                else:
                    aspects_dependencies[aspect_index][index_of_word2] = dependency[1]

    return aspects_dependencies


def process_sentence_dep(sentence):
    '''
    :param sentence: Sentence
    :return: Sentence with certain words removed
    '''
    sentence = remove_bert_numbering(sentence)
    # Removing contractions as the Standford NLP parser re-tokenizes words and it splits contractions.
    words = ['aint', '%', '-h', 'dont', '8pm', 'didnt', 'doesnt']
    for i in range(len(sentence)):
        sentence[i] = sentence[i].replace('#', '')
        if (sentence[i] in words):
            sentence[i] = 'removed'
    return sentence


def sentence_concatenate(sentence_left, sentence_target, sentence_right):
    '''
    :param sentence_left: left context
    :param sentence_target: target
    :param sentence_right: right context
    :return: concatenated sentence and aspect indices
    '''
    aspect_indices = []
    left_length = len(sentence_left)
    aspect_indices.append(list(range(left_length, left_length + len(sentence_target))))
    return sentence_left + sentence_target + sentence_right, aspect_indices


def make_dataset(layer_information, hypothesis_left, hypothesis_right, sen_length_left, sen_length_right,
                 corr_preds, info_left, info_right, aspect_hypo=False, randomize=True):
    '''

    :param layer_information: The layer information
    :param hypothesis_left: hypothesis for the words in the left context
    :param hypothesis_right: hypothesis for the words in the right context
    :param sen_length_left: left context length
    :param sen_length_right: right context length
    :param corr_preds: correctly predicted instances
    :param info_left: the layer being evaluated (left)
    :param info_right: the layer being evaluated (right)
    :param aspect_hypo: Boolean False
    :param randomize: Balance (True
    :return:The dataset
    '''
    x_corr = []
    x_incorr = []
    y_corr = []
    y_incorr = []
    cnt = 0
    if (aspect_hypo):
        changed_hypo_left = []
        changed_hypo_right = []
        for x in hypothesis_right:
            if x == 'no':
                changed_hypo_right.append([0, 1])
            else:
                changed_hypo_right.append([1, 0])
        for p in hypothesis_left:
            if p == 'no':
                changed_hypo_left.append([0, 1])
            else:
                changed_hypo_left.append([1, 0])
        hypothesis_right = changed_hypo_right
        hypothesis_left = changed_hypo_left

        # We change the hypothesis
    # Adding Left part first to make the dataset
    for i in range(len(sen_length_left)):
        # i= iterable for the number of sentences.
        if corr_preds[i] == 1:
            for j in range(sen_length_left[i]):
                # j=number of words in the left part
                # Iterating through all the words (j) then we move on to the next sentence
                x_corr.append(layer_information[info_left][0][i][j])
                y_corr.append(hypothesis_left[cnt])
                cnt = cnt + 1
        else:
            for j in range(sen_length_left[i]):
                x_incorr.append(layer_information[info_left][0][i][j])
                y_incorr.append(hypothesis_left[cnt])
                cnt = cnt + 1

    cnt = 0
    # Adding right part in a similar fashion to the right part
    for i in range(len(sen_length_right)):
        # i= iterable for the number of sentences.
        if corr_preds[i] == 1:
            for j in range(sen_length_right[i]):
                # j=number of words in the right part
                # Iterating through all the words (j) then we move on to the next sentence
                x_corr.append(layer_information[info_right][0][i][j])
                y_corr.append(hypothesis_right[cnt])
                cnt = cnt + 1
        else:
            for j in range(sen_length_right[i]):
                x_incorr.append(layer_information[info_right][0][i][j])
                y_incorr.append(hypothesis_right[cnt])
                cnt = cnt + 1
    if (randomize):
        #Based on the code by Lisa Meijer (https://github.com/lhmeijer/ABSCEM) to balance the dataset.
        class_size = calculate_classsize(np.asarray(y_corr))
        rand_indices = select_random_indices(class_size, np.asarray(y_corr))
        random_in = np.asarray(rand_indices)
        x_corr = np.asarray(x_corr)
        y_corr = np.asarray(y_corr)

        return x_corr[random_in], x_incorr, y_corr[rand_indices], y_incorr
    else:
        return np.asarray(x_corr), np.asarray(x_incorr), np.asarray(y_corr), np.asarray(y_incorr)


def make_dataset_test(layer_information, hypothesis_left, hypothesis_right, sen_length_left, sen_length_right,
                      info_left, info_right, aspect_hypo=False):
    '''
    Same as make_dataset but for test and hence it does not balance
    :param layer_information: The layer information
    :param hypothesis_left: hypothesis for the words in the left context
    :param hypothesis_right: hypothesis for the words in the right context
    :param sen_length_left: left context length
    :param sen_length_right: right context length
    :param info_left: the layer being evaluated (left)
    :param info_right: the layer being evaluated (right)
    :param aspect_hypo: Boolean False
    :return: the test dataset
    '''
    x = []
    y = []
    cnt = 0
    # Adding Left part first to make the dataset
    if (aspect_hypo):
        changed_hypo_left = []
        changed_hypo_right = []
        for t in hypothesis_right:
            if t == 'no':
                changed_hypo_right.append([0, 1])
            else:
                changed_hypo_right.append([1, 0])
        for p in hypothesis_left:
            if p == 'no':
                changed_hypo_left.append([0, 1])
            else:
                changed_hypo_left.append([1, 0])
        hypothesis_right = changed_hypo_right
        hypothesis_left = changed_hypo_left

    for i in range(len(sen_length_left)):
        # i= iterable for the number of sentences.
        for j in range(sen_length_left[i]):
            # j=number of words in the left part
            # Iterating through all the words (j) then we move on to the next sentence
            # if (j==sen_length_left[i]-1 and j <79):
            #   print(layer_information[info_left][0][i][j+1])
            x.append(layer_information[info_left][0][i][j])
            y.append(hypothesis_left[cnt])
            cnt = cnt + 1

    cnt = 0
    # Adding right part in a similar fashion to the right part
    for i in range(len(sen_length_right)):
        # i= iterable for the number of sentences.
        for j in range(sen_length_right[i]):
            # j=number of words in the right part
            # Iterating through all the words (j) then we move on to the next sentence
            x.append(layer_information[info_right][0][i][j])
            y.append(hypothesis_right[cnt])
            cnt = cnt + 1
    return np.asarray(x), np.asarray(y)


def format_hypothesis_pos_tag(hypothesis):
    if (hypothesis == [1, 0, 0, 0, 0]).all():
        return 0
    elif (hypothesis == [0, 1, 0, 0, 0]).all():
        return 1
    elif (hypothesis == [0, 0, 1, 0, 0]).all():
        return 2
    elif (hypothesis == [0, 0, 0, 1, 0]).all():
        return 3
    else:
        return 4

    return edited_hypothesis_corr


def get_information(hypothesis):
    '''
    Utility method to extract some information
    :param hypothesis:
    :return:
    '''
    cnt0 = 0
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0

    for i in hypothesis:
        if (format_hypothesis_pos_tag(i) == 0):
            cnt0 = cnt0 + 1
        elif format_hypothesis_pos_tag(i) == 1:
            cnt1 = cnt1 + 1
        elif format_hypothesis_pos_tag(i) == 2:
            cnt2 = cnt2 + 1
        elif format_hypothesis_pos_tag(i) == 3:
            cnt3 = cnt3 + 1
        else:
            cnt4 = cnt4 + 1

    return


def calculate_classsize(hypothesis):
    '''
    Code is from the work done by Lisa Meijer (https://github.com/lhmeijer/ABSCEM).
    :param hypothesis:
    :return:
    '''
    count = hypothesis.sum(axis=0)
    argmax = np.argmax(count)
    arr = np.arange(count.shape[0])
    edited_count = np.delete(arr, argmax)
    mean = int(np.floor(np.mean(count[edited_count])))
    class_size = np.zeros(count.shape, dtype=int)
    for i in range(count.shape[0]):
        class_size[i] = min(mean, count[i])
    return class_size


def select_random_indices(class_size, y):
    '''
    Code is from the work done by Lisa Meijer (https://github.com/lhmeijer/ABSCEM).
    :param class_size:
    :param y:
    :return:
    '''
    random_indices = []
    np.random.seed(0)
    for i in range(len(class_size)):
        a_range = np.arange(y.shape[0], dtype=int)
        y_arg_max = np.argmax(y, axis=1)  # getting the index of the max for each row.
        selected_items = a_range[y_arg_max == i]
        random_indices.append(
            np.random.choice(selected_items, class_size[i], replace=False).tolist())  # changed class_size[i] to 1500
    random_indices = [a for b in random_indices for a in b]
    return random_indices


def make_SentiAspect_dataset(Aspect_left, Aspect_right, sentiment_left, sentiment_right):
    '''

    :param Aspect_left:
    :param Aspect_right:
    :param sentiment_left:
    :param sentiment_right:
    :return:
    '''
    Aspect_senti_left = []
    Aspect_senti_right = []
    # making the left hypothesis
    for i in range(len(sentiment_left)):
        if Aspect_left[i] != 'no':
            # If it has some dependency then we check what the sentiment of the word is. It can just be neutral
            Aspect_senti_left.append(sentiment_left[i])
        else:
            # If its no, then that means we have no aspect sentiment so we allocate the same as neutral.
            Aspect_senti_left.append([0, 0, 1])
    # Making the right hypothesis
    for i in range(len(sentiment_right)):
        if Aspect_right[i] != 'no':
            Aspect_senti_right.append(sentiment_right[i])
        else:
            Aspect_senti_right.append([0, 0, 1])

    return Aspect_senti_left, Aspect_senti_right


def hyperparameter_optimization(x_train, y_train, x_test, y_test):
    '''

    :param x_train: train set
    :param y_train: train set label
    :param x_test: test set
    :param y_test: test set label
    :return: optimized accuracy
    '''
    parameter_space = {
        'hidden_layer_sizes': [(700,), (900,),(1100,)],
        'activation': ['relu'],
        'solver': ['adam'],
        'learning_rate': ['constant'],
    }
    print('Starting Gridsearch')
    mlp = MLPClassifier(max_iter=400)
    clf = GridSearchCV(mlp, parameter_space, n_jobs=1, cv=3).fit(x_train, y_train)
    print('Best parameters found:\n', clf.best_params_)
    py = clf.predict(x_test)

    argmax_pred = np.argmax(py, axis=1)
    argmax_true = np.argmax(y_test, axis=1)
    correct_score = 0
    cnt = 0
    for i in range(len(py)):
        if argmax_true[i] == argmax_pred[i]:
            correct_score = correct_score + 1
        cnt = cnt + 1

    hitrate = correct_score / cnt
    print(hitrate)
    return hitrate


def make_dataset_aspect(layer_information, hypothesis_left, hypothesis_right, sen_length_left, sen_length_right,
                        corr_preds, info_left, info_right, aspect_hypo=True, randomize=True):
    x_corr = []
    x_incorr = []
    y_corr = []
    y_incorr = []
    cnt = 0
    if (aspect_hypo):
        changed_hypo_left = []
        changed_hypo_right = []
        for x in hypothesis_right:
            if x == 'no':
                changed_hypo_right.append([0, 1])
            else:
                changed_hypo_right.append([1, 0])
        for p in hypothesis_left:
            if p == 'no':
                changed_hypo_left.append([0, 1])
            else:
                changed_hypo_left.append([1, 0])
        hypothesis_right = changed_hypo_right
        hypothesis_left = changed_hypo_left

        # We change the hypothesis
    # Adding Left part first to make the dataset
    for i in range(len(sen_length_left)):
        # i= iterable for the number of sentences.
        if corr_preds[i] == 1:
            for j in range(sen_length_left[i]):
                # j=number of words in the left part
                # Iterating through all the words (j) then we move on to the next sentence
                x_corr.append(layer_information[info_left][0][i][j])
                y_corr.append(hypothesis_left[cnt])
                cnt = cnt + 1
        else:
            for j in range(sen_length_left[i]):
                x_incorr.append(layer_information[info_left][0][i][j])
                y_incorr.append(hypothesis_left[cnt])
                cnt = cnt + 1

    cnt = 0
    # Adding right part in a similar fashion to the right part
    for i in range(len(sen_length_right)):
        # i= iterable for the number of sentences.
        if corr_preds[i] == 1:
            for j in range(sen_length_right[i]):
                # j=number of words in the right part
                # Iterating through all the words (j) then we move on to the next sentence
                x_corr.append(layer_information[info_right][0][i][j])
                y_corr.append(hypothesis_right[cnt])
                cnt = cnt + 1
        else:
            for j in range(sen_length_right[i]):
                x_incorr.append(layer_information[info_right][0][i][j])
                y_incorr.append(hypothesis_right[cnt])
                cnt = cnt + 1
    if (randomize):
        class_size = calculate_classsize(np.asarray(y_corr))
        rand_indices = select_random_indices(class_size, np.asarray(y_corr))
        random_in = np.asarray(rand_indices)
        x_corr = np.asarray(x_corr)
        y_corr = np.asarray(y_corr)

        return x_corr[random_in], x_incorr, y_corr[rand_indices], y_incorr
    else:
        return np.asarray(x_corr), np.asarray(x_incorr), np.asarray(y_corr), np.asarray(y_incorr)


def make_dataset_test_aspect(layer_information, hypothesis_left, hypothesis_right, sen_length_left, sen_length_right,
                             info_left, info_right, aspect_hypo=True):
    '''

    :param layer_information:
    :param hypothesis_left:
    :param hypothesis_right:
    :param sen_length_left:
    :param sen_length_right:
    :param info_left:
    :param info_right:
    :param aspect_hypo:
    :return:
    '''
    x = []
    y = []
    cnt = 0
    # Adding Left part first to make the dataset
    if (aspect_hypo):
        changed_hypo_left = []
        changed_hypo_right = []
        for t in hypothesis_right:
            if t == 'no':
                changed_hypo_right.append([0, 1])
            else:
                changed_hypo_right.append([1, 0])
        for p in hypothesis_left:
            if p == 'no':
                changed_hypo_left.append([0, 1])
            else:
                changed_hypo_left.append([1, 0])
        hypothesis_right = changed_hypo_right
        hypothesis_left = changed_hypo_left

    for i in range(len(sen_length_left)):
        # i= iterable for the number of sentences.
        for j in range(sen_length_left[i]):
            # j=number of words in the left part
            # Iterating through all the words (j) then we move on to the next sentence
            # if (j==sen_length_left[i]-1 and j <79):
            #   print(layer_information[info_left][0][i][j+1])
            x.append(layer_information[info_left][0][i][j])
            y.append(hypothesis_left[cnt])
            cnt = cnt + 1

    cnt = 0
    # Adding right part in a similar fashion to the right part
    for i in range(len(sen_length_right)):
        # i= iterable for the number of sentences.
        for j in range(sen_length_right[i]):
            # j=number of words in the right part
            # Iterating through all the words (j) then we move on to the next sentence
            x.append(layer_information[info_right][0][i][j])
            y.append(hypothesis_right[cnt])
            cnt = cnt + 1
    return np.asarray(x), np.asarray(y)


def edit_sentences(original_sentence):
    '''

    :param original_sentence:
    :return:
    '''
    edited_sentences = []
    lengths = []
    for sentence in original_sentence:
        new_sentence, new_sentence_len = combine_word_pieces_sentence(sentence)
        edited_sentences.append(new_sentence)
        lengths.append(new_sentence_len)
    return edited_sentences, lengths


def combine_word_pieces_sentence(sentence):
    '''
    :param sentence: Sentence
    :return: sentence with word pieces combined and length of the sentence
    '''
    sentence = remove_bert_numbering(sentence)
    edited_sentence = []
    flag = False
    for i in range(len(sentence)):
        if '##' in sentence[i]:
            if (i == 0):
                edited_sentence.append(sentence[i])
            else:
                last_word = edited_sentence.pop()
                cleaned_word = sentence[i].replace('#', '')
                combined_word = last_word + cleaned_word
                edited_sentence.append(combined_word)
        else:
            edited_sentence.append(sentence[i])
    return edited_sentence, len(edited_sentence)


def average_rows(prev_row, curr_row, num):
    '''

    :param prev_row: The row before the current row
    :param curr_row: the current row
    :param num: the number of word pieces that are being connected
    :return: the word embedding averaged.
    '''
    new_row = np.subtract(curr_row, prev_row)
    new_row = new_row / num
    average_row = prev_row + new_row
    return average_row


def combine_layer_information(layer_information, words_left, words_right, sen_lengths_left, sen_lengths_right):
    '''

    :param layer_information: layer information
    :param words_left: words in left context
    :param words_right: words in right context
    :param sen_lengths_left: length of left context
    :param sen_lengths_right: length of right context
    :return: combined layer information
    '''
    layer_information=reverse_hidden_layers_rights(layer_information,sen_lengths_right)
    edited_left_embedding, edited_right_embedding, edited_left_hidden_state, edited_right_hidden_state, edited_left_weighted_ini, edited_right_weighted_ini, \
    edited_left_weighted_0, edited_right_weighted_0, edited_left_weighted_1, edited_right_weighted_1 = [], [], [], [], [], [], [], [], [], []
    edited_words_left, edited_words_right = [], []
    # First do the left words.
    # left embeddings
    cnt = 0
    average_nums = 1
    for i in range(len(sen_lengths_left)):
        for j in range(sen_lengths_left[i]):
            word_l = words_left[cnt]
            if '##' in word_l:
                if j == 0:
                    # If j=0, that implies that we are at the beginning of a sentence and it has ##. So we should
                    # not do anything to this word
                    average_nums = 1
                    edited_words_left.append(word_l)
                    edited_left_embedding.append(layer_information['left_embedding'][0][i][j])
                    edited_left_hidden_state.append(layer_information['left_hidden_state'][0][i][j])
                    edited_left_weighted_ini.append(layer_information['left_weighted_state_ini'][0][i][j])
                    edited_left_weighted_0.append(layer_information['left_weighted_state_0'][0][i][j])
                    edited_left_weighted_1.append(layer_information['left_weighted_state_1'][0][i][j])
                else:
                    # Combining the words and the hidden layers. Taking their averages.
                    last_word = edited_words_left.pop()
                    cleaned_word = word_l.replace('#', '')
                    combined_word = last_word + cleaned_word
                    edited_words_left.append(combined_word)
                    average_nums = average_nums + 1
                    # Combining their word embeddings
                    last_embedding = edited_left_embedding.pop()
                    current_embedding = layer_information['left_embedding'][0][i][j]
                    average_embedding = average_rows(last_embedding, current_embedding, average_nums)
                    edited_left_embedding.append(average_embedding)
                    # Left_hidden_state
                    last_hidden_state = edited_left_hidden_state.pop()
                    current = layer_information['left_hidden_state'][0][i][j]
                    average = average_rows(last_hidden_state, current, average_nums)
                    edited_left_hidden_state.append(average)
                    # Left weighted_ini
                    last = edited_left_weighted_ini.pop()
                    current = layer_information['left_weighted_state_ini'][0][i][j]
                    average = average_rows(last, current, average_nums)
                    edited_left_weighted_ini.append(average)
                    # Left weighted 0
                    last = edited_left_weighted_0.pop()
                    current = layer_information['left_weighted_state_0'][0][i][j]
                    average = average_rows(last, current, average_nums)
                    edited_left_weighted_0.append(average)
                    # left_weighted 1
                    last = edited_left_weighted_1.pop()
                    current = layer_information['left_weighted_state_1'][0][i][j]
                    average = average_rows(last, current, average_nums)
                    edited_left_weighted_1.append(average)
            else:  # No need to concatenate the current word as it is not a word piece
                average_nums = 1
                edited_words_left.append(word_l)
                edited_left_embedding.append(layer_information['left_embedding'][0][i][j])
                edited_left_hidden_state.append(layer_information['left_hidden_state'][0][i][j])
                edited_left_weighted_ini.append(layer_information['left_weighted_state_ini'][0][i][j])
                edited_left_weighted_0.append(layer_information['left_weighted_state_0'][0][i][j])
                edited_left_weighted_1.append(layer_information['left_weighted_state_1'][0][i][j])
            cnt = cnt + 1
    # Right context
    cnt = 0
    average_nums = 1
    for i in range(len(sen_lengths_right)):
        for j in range(sen_lengths_right[i]):
            word_r = words_right[cnt]
            if '##' in word_r:
                if j == 0:
                    average_nums = 1
                    edited_words_right.append(word_r)
                    edited_right_embedding.append(layer_information['right_embedding'][0][i][j])
                    edited_right_hidden_state.append(layer_information['right_hidden_state'][0][i][j])
                    edited_right_weighted_ini.append(layer_information['right_weighted_state_ini'][0][i][j])
                    edited_right_weighted_0.append(layer_information['right_weighted_state_0'][0][i][j])
                    edited_right_weighted_1.append(layer_information['right_weighted_state_1'][0][i][j])
                else:
                    # Combining the words
                    last_word = edited_words_right.pop()
                    cleaned_word = word_r.replace('#', '')
                    combined_word = last_word + cleaned_word
                    edited_words_right.append(combined_word)
                    average_nums = average_nums + 1
                    # Combining their word embeddings
                    last_embedding = edited_right_embedding.pop()
                    current_embedding = layer_information['right_embedding'][0][i][j]
                    average_embedding = average_rows(last_embedding, current_embedding, average_nums)
                    edited_right_embedding.append(average_embedding)
                    # Left_hidden_state
                    last = edited_right_hidden_state.pop()
                    current = layer_information['right_hidden_state'][0][i][j]
                    average = average_rows(last, current, average_nums)
                    edited_right_hidden_state.append(average)
                    # Left weighted_ini
                    last = edited_right_weighted_ini.pop()
                    current = layer_information['right_weighted_state_ini'][0][i][j]
                    average = average_rows(last, current, average_nums)
                    edited_right_weighted_ini.append(average)
                    # Left weighted 0
                    last = edited_right_weighted_0.pop()
                    current = layer_information['right_weighted_state_0'][0][i][j]
                    average = average_rows(last, current, average_nums)
                    edited_right_weighted_0.append(average)
                    # left_weighted 1
                    last = edited_right_weighted_1.pop()
                    current = layer_information['right_weighted_state_1'][0][i][j]
                    average = average_rows(last, current, average_nums)
                    edited_right_weighted_1.append(average)

            else:  # No need to concatenate the current word
                average_nums = 1
                edited_words_right.append(word_r)
                edited_right_embedding.append(layer_information['right_embedding'][0][i][j])
                edited_right_hidden_state.append(layer_information['right_hidden_state'][0][i][j])
                edited_right_weighted_ini.append(layer_information['right_weighted_state_ini'][0][i][j])
                edited_right_weighted_0.append(layer_information['right_weighted_state_0'][0][i][j])
                edited_right_weighted_1.append(layer_information['right_weighted_state_1'][0][i][j])
            cnt = cnt + 1

    edited_layer_information = {
        'left_embedding': edited_left_embedding,
        'right_embedding': edited_right_embedding,
        'left_hidden_state': edited_left_hidden_state,
        'right_hidden_state': edited_right_hidden_state,
        'left_weighted_state_ini': edited_left_weighted_ini,
        'right_weighted_state_ini': edited_right_weighted_ini,
        'left_weighted_state_0': edited_left_weighted_0,
        'right_weighted_state_0': edited_right_weighted_0,
        'left_weighted_state_1': edited_left_weighted_1,
        'right_weighted_state_1': edited_right_weighted_1
    }
    return edited_layer_information, edited_words_left, edited_words_right


def make_dataset_edited(layer_information, hypothesis_left, hypothesis_right, sen_length_left, sen_length_right,
                        corr_preds, info_left, info_right, randomize=True):
    '''

    :param layer_information:
    :param hypothesis_left:
    :param hypothesis_right:
    :param sen_length_left:
    :param sen_length_right:
    :param corr_preds:
    :param info_left:
    :param info_right:
    :param randomize:
    :return:
    '''
    x_corr = []
    x_incorr = []
    y_corr = []
    y_incorr = []
    cnt = 0

    # Adding Left part first to make the dataset
    for i in range(len(sen_length_left)):
        # i= iterable for the number of sentences.
        if corr_preds[i] == 1:
            for j in range(sen_length_left[i]):
                # j=number of words in the left part
                # Iterating through all the words (j) then we move on to the next sentence
                x_corr.append(layer_information[info_left][cnt])
                y_corr.append(hypothesis_left[cnt])
                cnt = cnt + 1
        else:
            for j in range(sen_length_left[i]):
                x_incorr.append(layer_information[info_left][cnt])
                y_incorr.append(hypothesis_left[cnt])
                cnt = cnt + 1

    cnt = 0
    # Adding right part in a similar fashion to the right part
    for i in range(len(sen_length_right)):
        # i= iterable for the number of sentences.
        if corr_preds[i] == 1:
            for j in range(sen_length_right[i]):
                # j=number of words in the right part
                # Iterating through all the words (j) then we move on to the next sentence
                x_corr.append(layer_information[info_right][cnt])
                y_corr.append(hypothesis_right[cnt])
                cnt = cnt + 1
        else:
            for j in range(sen_length_right[i]):
                x_incorr.append(layer_information[info_right][cnt])
                y_incorr.append(hypothesis_right[cnt])
                cnt = cnt + 1

    if (randomize):
        class_size = calculate_classsize(np.asarray(y_corr))
        print(class_size)
        rand_indices = select_random_indices(class_size, np.asarray(y_corr))
        random_in = np.asarray(rand_indices)
        x_corr = np.asarray(x_corr)
        y_corr = np.asarray(y_corr)

        return x_corr[random_in], x_incorr, y_corr[rand_indices], y_incorr
    else:
        return np.asarray(x_corr), np.asarray(x_incorr), np.asarray(y_corr), np.asarray(y_incorr)


def make_dataset_test_edited(layer_information, hypothesis_left, hypothesis_right, sen_length_left, sen_length_right,
                             info_left, info_right, aspect_hypo=False):
    '''

    :param layer_information:
    :param hypothesis_left:
    :param hypothesis_right:
    :param sen_length_left:
    :param sen_length_right:
    :param info_left:
    :param info_right:
    :param aspect_hypo:
    :return:
    '''
    x = []
    y = []
    cnt = 0
    # Adding Left part first to make the dataset

    for i in range(len(sen_length_left)):
        # i= iterable for the number of sentences.
        for j in range(sen_length_left[i]):
            # j=number of words in the left part
            # Iterating through all the words (j) then we move on to the next sentence
            # if (j==sen_length_left[i]-1 and j <79):
            #   print(layer_information[info_left][0][i][j+1])
            x.append(layer_information[info_left][cnt])
            y.append(hypothesis_left[cnt])
            cnt = cnt + 1

    cnt = 0
    # Adding right part in a similar fashion to the right part
    for i in range(len(sen_length_right)):
        # i= iterable for the number of sentences.
        for j in range(sen_length_right[i]):
            # j=number of words in the right part
            # Iterating through all the words (j) then we move on to the next sentence
            x.append(layer_information[info_right][cnt])
            y.append(hypothesis_right[cnt])
            cnt = cnt + 1
    return np.asarray(x), np.asarray(y)

def reverse_hidden_layers_rights(layer_information, sentence_length_information_right):
    '''
    Reverse the right hidden layers
    :param layer_information: Layer information
    :param sentence_length_information_right: Sentence length of right contexts.
    :return: updated layer information with the right layer information reversed
    '''
    right_embedding='right_embedding'
    right_hidden_state='right_hidden_state'
    right_weighted_state_ini='right_weighted_state_ini'
    right_weighted_state_0='right_weighted_state_0'
    right_weighted_state_1='right_weighted_state_1'
    for i in range(len(sentence_length_information_right)):
        start_index=0
        copy_embedding=[]
        copy_hidden_state=[]
        copy_weighted_ini=[]
        copy_weighted_0=[]
        copy_weighted_1=[]
        for j in range(sentence_length_information_right[i]):
            #Copying the orginal
            copy_embedding.append(layer_information[right_embedding][0][i][j])
            copy_hidden_state.append(layer_information[right_hidden_state][0][i][j])
            copy_weighted_ini.append(layer_information[right_weighted_state_ini][0][i][j])
            copy_weighted_0.append(layer_information[right_weighted_state_0][0][i][j])
            copy_weighted_1.append(layer_information[right_weighted_state_1][0][i][j])

        for k in range(sentence_length_information_right[i]-1,-1,-1):
            layer_information[right_embedding][0][i][start_index]=copy_embedding[k]
            layer_information[right_hidden_state][0][i][start_index]=copy_hidden_state[k]
            layer_information[right_weighted_state_ini][0][i][start_index]=copy_weighted_ini[k]
            layer_information[right_weighted_state_0][0][i][start_index]=copy_weighted_0[k]
            layer_information[right_weighted_state_1][0][i][start_index]=copy_weighted_1[k]
            start_index=start_index+1

    return layer_information