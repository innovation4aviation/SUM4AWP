#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from collections import OrderedDict
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import preprocessing, get_vocab, get_token_pairs, get_matrix


def lexrank(raw_sent, word_embeddings, glove_dim):
    """
    Inspired by pagerank, lexrank considers each sentence as a node,
    give the weight to each edge by calculating cosine_similarity.
    And rank sentences by their score (which can represent the importance but also repeatability...).
    :param raw_sent: processed sentences in WPs.
    :param word_embeddings: word vectors by glove.
    :param glove_dim: the dimension of each word vector.
    :return: a list of ranked sentences(index 1) with their score(index 0).
    """
    num_sent = len(raw_sent)
    # preprocessing
    processed_sentences = preprocessing(raw_sent)
    # word and sentence representation
    sentence_vectors = []
    for i in processed_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((glove_dim,))) for w in i.split()]) / (len(i.split()) + 0.001)
        else:
            v = np.zeros((glove_dim,))
        sentence_vectors.append(v)
    # create a matrix
    sim_mat = np.zeros([num_sent, num_sent])
    # initial sim_mat by cosine_similarity
    for i in range(num_sent):
        for j in range(num_sent):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, glove_dim),
                                                  sentence_vectors[j].reshape(1, glove_dim))[0, 0]
    # turn to graph
    nx_graph = nx.from_numpy_array(sim_mat)
    # pagerank algorithm
    scores = nx.pagerank(nx_graph)
    # ranked sentence
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(raw_sent)), reverse=True)
    return ranked_sentences


def textrank_keywords(processed_sentences, window_size, top_num):
    """
    Inspired by pagerank, textrank considers each word as a node,
    give the weight to each edge by calculating word window pairs.
    And rank words by their score.
    :param processed_sentences: processed sentences, at least remove stopwords.
    :param window_size: the number of words following a word.
    :param top_num: the number of top words.
    :return: a list of Top top_num words (index 0) with their scores (index 1).
    """
    vocab = get_vocab(processed_sentences)
    token_pairs = get_token_pairs(window_size, processed_sentences)
    # Get normalized matrix
    g = get_matrix(vocab, token_pairs)
    # Initionlization for weight(pagerank value)
    pr = np.array([1] * len(vocab))
    d = 0.85  # damping coefficient, usually is .85
    min_diff = 1e-5  # convergence threshold
    steps = 10
    node_weight = None  # save keywords and its weight
    # Iteration
    previous_pr = 0
    for epoch in range(steps):
        pr = (1 - d) + d * np.dot(g, pr)
        if abs(previous_pr - sum(pr)) < min_diff:
            break
        else:
            previous_pr = sum(pr)
    # Get weight for each node
    node_weight = dict()
    for word, index in vocab.items():
        node_weight[word] = pr[index]
    # Print Top Keywords
    node_weight = OrderedDict(sorted(node_weight.items(), key=lambda t: t[1], reverse=True))
    keywords = []
    for i, (key, value) in enumerate(node_weight.items()):
        keywords.append((key, value))
        if i > (top_num-2):
            break
    return keywords
