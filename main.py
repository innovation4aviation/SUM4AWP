#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from models import lexrank, textrank_keywords
from utils import *


def lexrank4awps(pdf_path, word_embeddings, glove_dim, num_sent):
    """
    Sent one PDF, return a summary.
    :param pdf_path: the path of PDF.
    :param word_embeddings: words and their vectors.
    :param glove_dim: dimension of a word's vector.
    :param num_sent: the number of sentences you want as a summary.
    :return: a summary
    """
    # extract all content in PDF
    content = extract_pdf_plumber(pdf_path)
    # get cleaned main body text
    text = main_body_plumber(content)
    # sentence segmentation
    raw_sent = sent_segment(text)
    ranked_sent = lexrank(raw_sent, word_embeddings=word_embeddings, glove_dim=glove_dim)
    extract_summary = give_summary(ranked_sent, num_sent)
    return extract_summary


def textrank(pdf_path, window_size, top_num):
    """
    Sent one PDF, return key words.
    :param pdf_path: the path of PDF.
    :param window_size: the number of words following a word.
    :param top_num: the number of top words.
    :return: a list of Top top_num words
    """
    # extract all content in PDF
    content = extract_pdf_plumber(pdf_path)
    # get cleaned main body text
    text = main_body_plumber(content)
    # sentence segmentation
    raw_sent = sent_segment(text)
    # preprocessing
    processed_sentences = preprocessing(raw_sent)
    keywords = textrank_keywords(processed_sentences, window_size=window_size, top_num=top_num)
    # Return a list of keywords
    list_keywords = []
    for w in keywords:
        list_keywords.append(w[0])
    return list_keywords


def write_csv_content(folder_name, csv_name):
    """
    Write a csv with header 'WPID','SentenceRank' and 'Sentence'.
    :param folder_name: name of a folder where working papers are in.
    :param csv_name: the name of csv you will get.
    :return: a csv with header 'WPID','SentenceRank' and 'Sentence'.
    """
    num_pdf = 0
    pdfs = load_pdfs(folder_name)
    glove = 'glove/glove.6B.100d.txt'
    word_embeddings = load_glove(glove)
    with open(csv_name, 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        file_header = ['WPID', 'SentenceRank', 'Sentence']
        csv_writer.writerow(file_header)
        for pdf in pdfs:
            num_pdf += 1
            wpid = re.findall(r'(?<=wp_).*(?=_en)', pdf)
            content = extract_pdf_plumber(pdf)
            text = main_body_plumber(content)
            raw_sent = sent_segment(text)
            ranked_sent = lexrank(raw_sent, word_embeddings, glove_dim=100)
            rank = 0
            for sent in ranked_sent:
                rank += 1
                csv_writer.writerow([wpid[0], rank, sent[1]])
    print("-" * 30 + "\nReport:\n" + str(num_pdf) + " extracted.\n")


def write_csv_wps(folder_name, csv_name):
    """
    Write a csv with header 'WPID','AgendaItem','Title','Action'.
    :param folder_name: name of a folder where working papers are in.
    :param csv_name: the name of csv you will get.
    :return: a csv with header 'WPID','AgendaItem','Title','Action'.
    """
    num_pdf = 0
    pdfs = load_pdfs(folder_name)
    with open(csv_name, 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        file_header = ['WPID', 'AgendaItem', 'Title', 'Action']
        csv_writer.writerow(file_header)
        for pdf in pdfs:
            num_pdf += 1
            wpid = re.findall(r'(?<=wp_).*(?=_en)', pdf)
            content_plumber = extract_pdf_plumber(pdf)
            try:
                agenda_num, agenda_item, title = get_opening(content_plumber)
                summary, action = get_summary_plumber(content_plumber)
                csv_writer.writerow([wpid[0], agenda_num, title, action.replace(';', ',')])
            except:
                csv_writer.writerow([wpid, '', '', action.replace(';', ',')])
    print("-" * 30 + "\nReport:\n" + str(num_pdf) + " extracted.\n")


def write_csv_keywords(folder_name, csv_name):
    """
    Write a csv with header 'WPID','KeywordRank' and 'Keyword'.
    :param folder_name: name of a folder where working papers are in.
    :param csv_name: the name of csv you will get.
    :return: a csv with header 'WPID','KeywordRank' and 'Keyword'.
    """
    num_pdf = 0
    pdfs = load_pdfs(folder_name)
    with open(csv_name, 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        file_header = ['WPID', 'KeywordRank', 'Keyword']
        csv_writer.writerow(file_header)
        for pdf in pdfs:
            num_pdf += 1
            wpid = re.findall(r'(?<=wp_).*(?=_en)', pdf)
            keywords = textrank(pdf, window_size=4, top_num=30)
            rank = 0
            for w in keywords:
                rank += 1
                csv_writer.writerow([wpid[0], rank, w])
    print("-" * 30 + "\nReport:\n" + str(num_pdf) + " extracted.\n")


def write_dataset(folder_name, csv_name):
    """
    Write a csv with header 'WPID','Summary' and 'Text' (main body).
    :param folder_name: name of a folder where working papers are in.
    :param csv_name: the name of csv you will get.
    :return: a csv with header 'WPID','Summary' and 'Text'.
    """
    num_pdf = 0
    pdfs = load_pdfs(folder_name)
    with open(csv_name, 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        file_header = ['WPID', 'Summary', 'Text']
        csv_writer.writerow(file_header)
        for pdf in pdfs:
            num_pdf += 1
            wpid = re.findall(r'(?<=wp_).*(?=_en)', pdf)
            content = extract_pdf_plumber(pdf)
            # get cleaned main body text
            text = main_body_plumber(content)
            summary, action = get_summary_plumber(content)
            csv_writer.writerow([wpid[0], summary.replace(';', ','), text])
    print("-" * 30 + "\nReport:\n" + str(num_pdf) + " extracted.\n")
