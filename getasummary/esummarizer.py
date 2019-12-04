import numpy as np
import pandas as pd
import re
import networkx as nx
import requests
from bs4 import BeautifulSoup
from rouge import Rouge
import codecs
import PyPDF2

from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import nltk

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.spatial.distance import cosine

#location of GloVe file
embedding_path = './glove/glove.6B.100d'

def readFile(file_name):
    # open the file as read only
    # Return all the text in the file (in parapgraphs)
    with open(file_name) as file:
        text = [line.strip('\n') for line in file.readlines() if line.strip()]
    # close the file
    file.close()
    return ' '.join(text)

#added support for PDF file 2/12/2019
def readPDF(file_name):
    text = []
    pdf_file_object = open(file_name, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdf_file_object)
    numPages = pdfReader.numPages
    print("No. of pages: ",numPages)
    for i in range(numPages):
        page_object = pdfReader.getPage(i).extractText().replace("\n","")
        text.append(page_object)
        print("length of page: ",len(page_object))
    return text

def get_wordcount(text):
    # using regex (findall()) to count words in text
    wc = len(re.findall(r'\w+', text))
    return(wc)

def get_sentencecount(text):
    # count no. of sentences in text
    return len(sent_tokenize(text))

def readUrl(url):

    #return the text of the article
    res = requests.get(url)
    res.status_code
    print(res.status_code)
    if res.status_code == 200:
        #create a beautifulsoup object
        soup = BeautifulSoup(res.content, 'lxml')
        text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
        return text
    else:
        print('Error assessing page, status = ' + str(res.status_code))

def get_sentences(text):

    sentences = sent_tokenize(text)

    return sentences

stop_words = set(stopwords.words('english'))
def clean_sentence(sentence,stop_words):
    # input : a sentence
    # output : a cleaned sentence

    #convert to lowercase
    new_sen = sentence.lower()

    #remove xml
    new_sen = BeautifulSoup(new_sen, 'lxml').text

    #remove 's', punctuation, special characters, text in ()
    new_sen = re.sub(r'\([^)]*\)', '', new_sen)
    new_sen = re.sub('"','', new_sen)
    new_sen = re.sub("\n", "", new_sen)
    new_sen = re.sub(r"'s\b","",new_sen)
    new_sen = re.sub("[^a-zA-Z]", " ", new_sen)

    #remove stop_words
    tokens = [w for w in new_sen.split() if not w in stop_words]

    #remove short words
    long_words=[]
    for i in tokens:
        if len(i)>=3:
            long_words.append(i)
    return (' '.join(long_words))

def clean_all_sentences(sentences,stop_words):
    # input: list of sentences
    # output: list of cleaned sentences
    cleaned = []
    for s in sentences:
        s = clean_sentence(s,stop_words)
        cleaned.append(''.join(s))
    # remove empty sentence
    cleaned = [s for s in cleaned if len(s) > 0]
    return(cleaned)

#------------------------------------------------------------------------------
##faster way to load glove. split it into .vocab, .npy
def convert_to_binary(embedding_path):
    """
    Here, it takes path to embedding text file provided by glove.
    :param embedding_path: takes path of the embedding which is in text format or any format other than binary.
    :return: a binary file of the given embeddings which takes a lot less time to load.
    """
    f = codecs.open(embedding_path + ".txt", 'r', encoding='utf-8')
    wv = []
    with codecs.open(embedding_path + ".vocab", "w", encoding='utf-8') as vocab_write:
        count = 0
        for line in f:
            if count == 0:
                pass
            else:
                splitlines = line.split()
                vocab_write.write(splitlines[0].strip())
                vocab_write.write("\n")
                wv.append([float(val) for val in splitlines[1:]])
            count += 1
    np.save(embedding_path + ".npy", np.array(wv))

def load_embeddings_binary(embedding_path):
    """
    It loads embedding provided by glove which is saved as binary file. Loading of this model is
    about  second faster than that of loading of txt glove file as model.
    :param embeddings_path: path of glove file.
    :return: glove model
    """
    with codecs.open(embedding_path + '.vocab', 'r', 'utf-8') as f_in:
        index2word = [line.strip() for line in f_in]
    wv = np.load(embedding_path + '.npy')
    word_embeddings = {}
    for i, w in enumerate(index2word):
        word_embeddings[w] = wv[i]
    return word_embeddings
#-------------------------------------------------------------------------------

def create_sentence_vectors(sentences, word_embeddings):
    #input : list of sentences
    #output  list of sentences vectors
    #sentence vectors = average of (sum of word vectors in that sentence/length of sentence)
    sentence_vectors = []
    for i in sentences:
        if len(i) != 0:
              v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    return(sentence_vectors)

def get_similarity_matrix(sentences,sentence_vectors):
    sim_mat = np.zeros([len(sentences), len(sentences)])

    for i in range(len(sentences)):
      for j in range(len(sentences)):
        if i != j:
          sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

    return(sim_mat)

# Convert matrix to graph
def convert_matrix_to_graph(sim_mat):
    nx_graph = nx.from_numpy_array(sim_mat)
    return(nx_graph)

# Apply TextRank Algorithm to score the sentence
# The more similar a sentence is with all other sentences, the higher the score
def score_sentences(nx_graph):
    scores = nx.pagerank(nx_graph)
    return scores

def rank_sentences(sentences,scores):

    ranked_sentences = sorted(((scores[i],s,i) for i,s in enumerate(sentences)),
                          reverse=True)
    return ranked_sentences

# Extract top_n sentences as the summary,
# input :
#         ranked_sentneces: list of sentences with its score
#         top_n : number of sentences to extract
#         rank : whether the output will be sorted by rank order, default: False
# Output :
#         return list of top sentences
def extract_top_sentences(ranked_sentences,top_n, rank=False):

    if rank == True:
        s = sorted(ranked_sentences[0:top_n],key=lambda x: x[0])
    else:
        s = sorted(ranked_sentences[0:top_n],key=lambda x: x[2])
    return [s[1] for s in s]

def extract_top_sentences_sim_threshold(ranked_sentences,top_n, sim_mat,rank=False, sim_threshold = 0.95):

    sentence_scores_sort = ranked_sentences
    sentences_summary = []
    count = 1

    for s in sentence_scores_sort:

        if count > top_n:
            break
        include_flag = True
        for ps in sentences_summary:
            #sim = similarity(sim_mat[s[2]], sim_mat[ps[2]])
            sim = sim_mat[s[2]][ps[2]]

            # exclude similar sentences based on sim_threshold
            if sim > sim_threshold:
                include_flag = False
        if include_flag:
            sentences_summary.append(s)
            count += 1

    if rank == True:
        s = sentences_summary
    else:
        s = sorted(sentences_summary,key=lambda x: x[2])
    return [s[1] for s in s]

#--------------------------------------------------------------
# Graph-Based Summarization using TextRank
#--------------------------------------------------------------
def generate_summary_textrank(text,top_n,word_embeddings):

    # Get list of sentences
    sentences = get_sentences(text)

    # clean all sentences
    print('cleaning sentences ...')
    sentences_c = clean_all_sentences(sentences,stop_words)

    # Create word vectors for cleaned sentences
    print('creating word vectors for sentences ...')
    sentence_vectors = create_sentence_vectors(sentences_c,word_embeddings)

    # Get Sentence Similiarity Matrix
    print('creating sentence similarity matrix ...')
    sim_mat = get_similarity_matrix(sentences_c,sentence_vectors)

    # Convert matrix to graph and score sentences
    print('scoring sentences ...')
    nx_graph = convert_matrix_to_graph(sim_mat)
    scores = score_sentences(nx_graph)

    # Rank the sentences
    print('ranking sentences ...')
    ranked_sentences = rank_sentences(sentences,scores)

    # Extract top sentences & generate summary
    print('generating summary ...')
    #generated_summary = extract_top_sentences(ranked_sentences,top_n)

    # apply similarity threshold in selection 26/11/2019
    generated_summary = extract_top_sentences_sim_threshold(ranked_sentences,top_n, sim_mat, rank=False, sim_threshold = 0.92)


    generated_summary = "\n".join(generated_summary)
    print('completed')
    return (generated_summary)

    #--------------------------------------------------------------
    # Centroid-Based Summarization using BOW
    #--------------------------------------------------------------

topic_threshold=0.3
sim_threshold=0.95

def similarity(v1, v2):
    score = 0.0
    if np.count_nonzero(v1) != 0 and np.count_nonzero(v2) != 0:
        score = ((1 - cosine(v1, v2)) + 1) / 2
    return score

def generate_summary_cbow(text, top_n):

    topic_threshold=0.3
    sim_threshold=0.95

    # Get list of sentences
    sentences = get_sentences(text)

    # clean all sentences
    print('cleaning sentences ...')
    sentences_c = clean_all_sentences(sentences,stop_words)
    #print(sentences_c)

    # convert sentences to word vectors using countvectorizer
    print('converting to TFIDF ...')
    vectorizer = CountVectorizer()
    sent_word_matrix = vectorizer.fit_transform(sentences_c)

    # transform word vectors to tfidf
    transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
    tfidf = transformer.fit_transform(sent_word_matrix)
    tfidf = tfidf.toarray()


    # find centroid vector
    # for sentence vector that is less than topic_threshold, set it to 0
    print('getting sentences close to centroid vector ...')
    centroid_vector = tfidf.sum(0)
    centroid_vector = np.divide(centroid_vector, centroid_vector.max())
    for i in range(centroid_vector.shape[0]):
        if centroid_vector[i] <= topic_threshold:
            centroid_vector[i] = 0

    # sentence scoring based on similarity with the centroid vector
    print('Scoring sentences ...')
    sentences_scores = []
    for i in range(tfidf.shape[0]):
        score = similarity(tfidf[i, :], centroid_vector)
        sentences_scores.append((i, sentences[i], score, tfidf[i, :]))

    sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True)
    #print(sentence_scores_sort)

    # sentence selection
    print('Extracting sentences ...')
    count = 1
    sentences_summary = []
    for s in sentence_scores_sort:
        if count > top_n:
            break
        include_flag = True
        for ps in sentences_summary:
            sim = similarity(s[3], ps[3])

            # exclude similar sentences based on sim_threshold
            if sim > sim_threshold:
                include_flag = False
        if include_flag:
            sentences_summary.append(s)
            count += 1

    # sort the summary based on sentence order
    sentences_summary_sort = sorted(sentences_summary, key=lambda el: el[0])

    generated_summary = "\n".join([s[1] for s in sentences_summary_sort])
    print('Completed ...')
    print(generated_summary)
    return generated_summary



    #---------------------------------------------------------------------------
    # Pre-Trained Bert Summarizer (Not deployed due to memory limitation in aws)
    #---------------------------------------------------------------------------
#from summarizer import Summarizer
#
#def generate_summary_bert(text,sent_count, num_sentences_to_match):
#    bert_model = Summarizer()
#    generated_summary = bert_model(text,num_sentences_to_match/sent_count)
#    return(generated_summary)
