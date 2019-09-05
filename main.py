#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from models import lexrank
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
            wpid = re.findall(r'\d+(?=_en)', pdf)
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
            wpid = re.findall(r'\d+(?=_en)', pdf)
            content_plumber = extract_pdf_plumber(pdf)
            agenda_num, agenda_item, title = get_opening(content_plumber)
            summary, action = get_summary_plumber(content_plumber)
            csv_writer.writerow([wpid[0], agenda_num, title, action.replace(';', '，')])
    print("-" * 30 + "\nReport:\n" + str(num_pdf) + " extracted.\n")
