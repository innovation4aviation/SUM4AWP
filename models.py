#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import preprocessing


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
