#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import glob
from io import StringIO
import matplotlib.pyplot as plt
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer
import numpy as np
import pandas as pd
# PDFMiner (https://github.com/pdfminer/pdfminer.six)
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
# pdfplumber (https://github.com/jsvine/pdfplumber/)
# Does not work with working papers before A35
import pdfplumber
import re
from wordcloud import WordCloud


def load_pdfs(folder_name):
    """
    Load all pdfs in one folder.
    :param folder_name: the name of folder which you need.
    :return: all pdf in the folder which is input.
    """
    pdf_path = folder_name + '/'
    pdfs = glob.glob("{}/*.pdf".format(pdf_path))
    return pdfs


def extract_pdf_plumber(pdf_path):
    """
    Extract All Content in Working Paper by pdfplumber.
    https://github.com/jsvine/pdfplumber/
    Better than extract_pdf_content.
    But works only with working papers after A35.
    :param pdf_path: the path of pdf
    :return: all the content
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            content_plumber = ''
            for page in pdf.pages:
                content_plumber += str(page.extract_text())
        return content_plumber
    except:
        return '< PDF Error >'


def get_opening(content_plumber):
    """
    Extract Agenda Item and TITLE from extracted content.
    :param content_plumber: extracted content from PDF of working paper by pdfplumber
    :return: number of agenda item, agenda item and title
    """
    opening = re.findall(r'\n *Agenda Item.*(?=\n *\(Presented )|'
                         r'\n *Agenda Item.*(?=\nEXECUTIVE SUMMARY)|'
                         r'\n *Agenda Item.*(?=\nCORRIGENDUM)', content_plumber, flags=re.S)
    agenda_num = re.findall(r'(?<=Agenda Item[: ])[A-Z]?[\d ]+', opening[0])
    agenda_item = re.sub(r'\n *\n[A-Z-,/’:]+ .*$', '', opening[0], flags=re.S)
    title = re.sub(r'Agenda Item.*(?=\n *\n[A-Z-,/’:]+ )', '', opening[0], flags=re.S)
    return ''.join(agenda_num[0].split()), ' '.join(agenda_item.split()), ' '.join(title.split())


def get_summary_plumber(content_plumber):
    """
    Extract SUMMARY and Action from extracted content.
    :param content_plumber: extracted content from PDF of working paper by pdfplumber
    :return: summary and action of working paper
    """
    summary_box = re.findall(r'(?<=[\n ]EXECUTIVE SUMMARY ).*?(?=\nStrategic )|'
                             r'(?<=[\n ]EXECUTIVE SUMMARY ).*?(?=\nReferences: )', content_plumber, flags=re.S)
    if summary_box != []:  # Cases with SUMMARY
        # In case the SUMMARY is over 1 page, remove footnote and header
        clean_summary_box = re.sub(r'\n *\n\d +[A-Zhw‘\n].*?(?=A\d+-WP/\d+ )|'
                                   r' *A\d+-WP/\d+ *\n*(- ?\d+ ?-)* *\n*[A-Z]+/\d+(, [A-Z]+/\d+)* *(- ?\d+ ?-)* *\n|'
                                   r'- ?\d+ ?-', '', summary_box[0], re.VERBOSE, flags=re.S)
        # Cases with or without summary
        summary = re.sub(r'Action: .*$|Action required:.*$', '', clean_summary_box, flags=re.S)
        # Cases with or without action
        if "Action:" in clean_summary_box or "Action required:" in clean_summary_box:  # Cases with action
            action = re.sub(r'.*Action:|.*Action required:', '', clean_summary_box, flags=re.S)
        else:  # Cases without action
            action = ''
    else:  # Case 1: no SUMMARY, then no summary, no action
        summary = ''
        action = ''
    return ' '.join(summary.split()), ' '.join(action.split())


def main_body_plumber(content_plumber):
    """
    Extract the main body from extracted content.
    :param content_plumber: extracted content from PDF of working paper by pdfplumber
    :return: main body of working paper without opening, summary, appendix, annex, etc.
    """
    # remove appendix
    content_without_appen = re.sub(r'(— ?){2,}.*$|(- ?){2,}.*$|APPENDIX.*$|'
                                   r'[-—]+ END [-—]+.*$', '\n', content_plumber, re.VERBOSE, flags=re.S)
    # remove content before main body
    content_main_body = re.sub(r'.*\n *1\. +|.*\nReferences: |.*\n\(Presented[A-Za-z ]+\)|'
                               r'.*\nCORRIGENDUM', '\n1. ', content_without_appen, re.VERBOSE, flags=re.S)
    # remove footnote, header, page number, end,
    main_body = re.sub(r"\n *\n\d +[A-Zhw‘\n].*?(?=A\d+-WP/\d+ )|"  # footnote 
                       r" *A\d+-WP/\d+ *\n*(- ?\d+ ?-)* *\n*[A-Z]+/\d+(, [A-Z]+/\d+)* *(- ?\d+ ?-)* *\n|"  # header 
                       r"- ?\d+ ?-|[-–—]+ ?END ?[-–—]+|(— ?){2,}", '\n', content_main_body, re.VERBOSE, flags=re.S)
    # remove outlines and their TITLEs
    main_body_sent = re.sub(r"- ?\d+ ?-|[-–—]+ ?END ?[-–—]+|(— ?){2,}|\n\d\. +[A-Z,\d’/\n\-– :()]+ |"  # TITLEs
                            r"\n\d\.\d +[A-Za-z0-9 -]{,70}(?=\n)", '\n', main_body, re.VERBOSE, flags=re.S)
    # remove bullets, unicode?
    main_body_rm_bulltes = re.sub(r'\\uf0[a-z0-9]\d|\n\d+(\.\d+)+\.? +|'
                                  r'\n― |\n[a-z]\)|\n• |\n\d\. ', '\n', main_body_sent, re.VERBOSE, flags=re.S)
    # get clean text
    text_main_body = ' '.join(main_body_rm_bulltes.split())
    return text_main_body.replace(';', ',')  # use .replace(';', '.') if you do not like long sentences with ';'


def extract_pdf_content(pdf):
    """
    Extract All Content in Working Paper by pdfminer.
    https://github.com/pdfminer/pdfminer.six
    Attention: this function can extract all pdf, but may occur disorder of the phrase in on sentence.
    It is better to abandon this function.
    :param pdf: the path of pdf
    :return: all the content
    """
    rsrcmgr = PDFResourceManager()
    codec = 'utf-8'
    outfp = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr=rsrcmgr, outfp=outfp, codec=codec, laparams=laparams)
    with open(pdf, 'rb') as fp:
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()
        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,
                                      caching=caching, check_extractable=True):
            interpreter.process_page(page)
    content = outfp.getvalue()
    device.close()
    outfp.close()
    return content


def get_main_body(content):
    """
    Extract the main body from extracted content. Based on extracting content by pdfminer.
    Attention: It is better to abandon pdfminer.
    :param content: contend extracted by pdfminer (function extract_pdf_content())
    :return: main body of working paper without opening, summary, appendix, annex, etc.
    """
    # remove appendix
    content_without_appen = re.sub(r'(— ?){2,}.*$|\nAPPENDIX.*$|— END —.*$|[\f\r\t\v]', '\n', content, flags=re.S)
    # remove content before main body
    content_main_body = re.sub(r'.*\n1\. +', '\n', content_without_appen, flags=re.S)
    # remove footnote, header, page number, end, outlines
    main_body = re.sub(r"\n\d +[A-Zhw].*?(?=A\d+-WP/\d+ \n)|A\d+-WP/\d+ +\n[A-Z]+/\d+|- \d+ ?-|"
                       r"—+ END —+|(— ?){2,}|-+ ?END ?-+|\n\d+\. +|–+ END –+|"
                       r"\n *\d+(\.\d+\.?)* *\n|\n\d. +[A-Z]+ \n", '\n', content_main_body, re.VERBOSE, flags=re.S)
    # remove TITLES of outlines
    main_body_sent = re.sub(r'\n[A-Z,\d’/\n- :()]+\n', '\n', main_body, flags=re.S)
    # remove bullets
    main_body_rm_bulltes = re.sub(r'\n― |\n[a-z]\)|\n• ', '\n', main_body_sent, flags=re.S)
    # remove \f, \n, \r, \t, \v
    clean_main_body = re.sub(r'[\n]', '', main_body_rm_bulltes)
    # get clean text
    text_main_body = ' '.join(clean_main_body.split())
    return text_main_body.replace(';', ',')  # use .replace(';', '.') if you do not like long sentences with ';'


def sent_segment(text):
    """Sentence segmentation avoiding the abbreviation."""
    punkt_param = PunktParameters()
    abbreviation = ['i.e', 'e.g', 'U.S', 'Dr', 'No', 'etc', 'Note', 'Vol', 'Ref', 'para', 'NO']
    punkt_param.abbrev_types = set(abbreviation)
    tokenizer = PunktSentenceTokenizer(punkt_param)
    raw_sentences = tokenizer.tokenize(text)
    return raw_sentences


def remove_stopwords(sen):
    """Remove 179 stop words which are frequent but not important."""
    # load 179 stop words in nltk.corpus
    stop_words = stopwords.words('english')
    # remove stop words
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


def preprocessing(raw_sentences):
    """Only leave the words, and lower words, remove 179 stop words."""
    # only keep words
    clean_sentences = pd.Series(raw_sentences).str.replace('[^a-zA-Z]', ' ')
    # lower words
    lower_sentences = [s.lower() for s in clean_sentences]
    # remove 179 stop words
    processed_sentences = [remove_stopwords(r.split()) for r in lower_sentences]
    return processed_sentences


def load_glove(glove):
    """Load words and their vectors trained by GloVe (https://nlp.stanford.edu/projects/glove/)."""
    word_embeddings = {}
    with open(glove, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs
    return word_embeddings


def draw_heatmap(sim_mat):
    """Draw a heatmap to show the sentence similarity matrix."""
    fig = plt.figure(dpi=400)
    x_label = range(len(sim_mat))
    y_label = range(len(sim_mat))
    ax = fig.add_subplot(111)
    ax.set_yticks(range(len(y_label)))
    ax.set_yticklabels(y_label, fontsize=4)
    ax.set_xticks(range(len(x_label)))
    ax.set_xticklabels(x_label, fontsize=3)
    im = ax.imshow(sim_mat, cmap=plt.cm.Blues)
    plt.colorbar(im)
    plt.title("Sentence Similarity Matrix", fontsize=9)
    plt.show()


def draw_hist(sim_mat):
    """Draw a histogram to show the distribution of the sentence similarity."""
    fig = plt.figure(dpi=150)
    data = sim_mat.flatten()  # turn into 1d array
    bins = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
            0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    # bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.hist(data, bins, density=0, facecolor="blue", edgecolor="black", alpha=0.6)
    plt.xlabel("Sentence Similarity")
    plt.ylabel("Pair Amount")
    plt.title("Sentence Similarity Distributions", fontsize=10)
    plt.show()


def draw_graph(nx_graph, base_weight):
    """To show the sentences which are similar more than the score you give."""
    fig = plt.figure(dpi=200)
    elarge = [(u, v) for (u, v, d) in nx_graph.edges(data=True) if d['weight'] > base_weight]
    nx.draw(nx_graph,
            pos=nx.shell_layout(nx_graph),  # random_layout, circular_layout, spring_layout, spectral_layout
            node_size=120, node_color='black',
            node_shape='o', alpha=0.7,
            edgelist=elarge, width=0.9,
            edge_color='grey', style='solid',
            with_labels=True,
            font_size=8, font_weight='bold', font_color='w')
    plt.title("Sentence Similarity More Than " + str(base_weight), fontsize=10)
    plt.show()


def draw_sub_graph(nx_graph, base_weight):
    """To show the sentences which are similar more than the score you give in a random way."""
    sub_graph = []
    for (u, v, d) in nx_graph.edges(data=True):
        if d['weight'] > base_weight:
            sub_graph.append((u, v))
    nx_sub = nx.Graph(sub_graph)
    fig = plt.figure(dpi=200)
    node_size = [nx_sub.degree(u)*15+150*base_weight for u in nx_sub]
    node_color = [float(nx_sub.degree(u)) for u in nx_sub]
    nx.draw(nx_sub,
            pos=nx.spring_layout(nx_sub),  # random_layout, circular_layout, shell_layout, spectral_layout
            node_size=node_size, node_color=node_color,
            cmap=plt.cm.twilight_shifted_r,
            node_shape='o', alpha=0.8,
            width=0.9,
            edge_color='grey', style='solid',
            with_labels=True,
            font_size=8, font_weight='bold', font_color='white')
    plt.title("Sentence Similarity More Than " + str(base_weight), fontsize=10)
    plt.show()


def give_summary(ranked_sentences, num):
    """
    Regroup the ranked sentences as a summary.
    :param ranked_sentences: ranked sentences after employing lexrank.
    :param num: the number of sentences you want in the summary.
    :return: a summary
    """
    extract_summary = ''
    for i in range(num):  # same sentences amount with reference sent_num or top 5 or 10
        extract_summary += ranked_sentences[i][1] + ' '
    return extract_summary


def get_wordcloud(text):
    """
    Show the frequent and important words in the corpus.
    :param text: the corpus in one string.
    :return: a figure to show the frequent and important words in the corpus.
    """
    wordcloud = WordCloud(background_color='white', collocations=False, width=400, height=300, margin=5).generate(text)
    fig = plt.figure(dpi=250)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
