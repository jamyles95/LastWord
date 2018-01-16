# Text Mining 
# import gensim 
import nltk
# nltk.download('punkt')
# from nltk.corpus import words
# from nltk.corpus import brown
# nltk.download('words')
# nltk.download('brown')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')
import re
import string
import numpy as np
import math
import copy
import csv
import pandas as pd
from sentiment_module import sentiment


tau = 0.6
k = 40
z = 10
max_words = 10

# -----------------------------------------------------------------------
## Functions

def norm_mat_fn(old_matrix,axis):
    terms, docs = np.shape(old_matrix)
    norm = np.apply_along_axis(np.linalg.norm, axis=axis, arr=old_matrix)
    norm_matrix = np.zeros((terms,docs))    
    if (len(np.where(norm == 0)[0]) != 0):
        for n in range(0,docs):
            if (n in np.where(norm == 0)[0]):
                norm_matrix[:,n] = 0
            else:
                norm_matrix[:,n] = old_matrix[:,n] / norm[n] 
    else:
        norm_matrix = old_matrix/norm
    return(norm_matrix)

def max_words_fn(words_list, cluster_list, num_of_words):
    max_words = []
    for i in range(0,len(cluster_list[0,:])):
        max_words_temp = []
        for j in range(num_of_words*-1,0):
            max_words_temp.insert(0,words_list[cluster_list[:,i].argsort()[j]])
        max_words.append(max_words_temp)
    return(max_words)


def unique_term_shrink_fn(term_freq_mat,old_unique_terms, min_docs,docs):
    max_val = math.log(docs,min_docs)   
    sum_mat = np.zeros(docs)
    sum_mat = term_freq_mat.sum(axis = 1)
    unique_terms = []
    for i in np.where(sum_mat < max_val )[0]:
        unique_terms.append(old_unique_terms[i])    
    return(unique_terms)


def TF_IDF_fn(list_term_vec,unique_terms):
    terms = len(unique_terms)
    docs = len(list_term_vec)
    tf = np.zeros((terms,docs))
    idf = np.zeros((terms,1))
    w = np.zeros((terms,docs)) 
    for n in range(0,docs):
        for m in range(0,terms):
            tf[m,n] = list_term_vec[n].count(unique_terms[m])
    term_counts = np.apply_along_axis(np.count_nonzero, axis=1, arr=tf)
    for m in range(0,terms):
        idf[m] = math.log(docs/term_counts[m])
    for n in range(0,docs):
        for m in range(0,terms):
            w[m,n] = tf[m,n]*idf[m]
    w_norm = np.apply_along_axis(np.linalg.norm, axis=0, arr=w)
    return(tf,idf,w,w_norm)


def LSA_mat_fn(tfidf, k):
    U,s,Vt = np.linalg.svd(tfidf, full_matrices=True)
    S = np.diag(s)
    U_k = U[:,0:k]
    S_k = S[0:k,0:k]
    Vt_k = Vt[0:k,:]
    X_k = np.dot(np.dot(U_k,S_k),Vt_k)
    return(s,X_k)


def MST_tau_fn(compare_docs, tau):
    delta_mat = 1 - compare_docs
    docs = len(delta_mat)
    E_mat = np.empty((docs,docs)) * np.nan
    for n in range(0,docs):
        for n2 in range(0,docs):
            E_mat[n,n2] = delta_mat[n,n2]
    e_min_row, e_min_col = np.unravel_index(np.nanargmin(E_mat),E_mat.shape)
    E_new = np.matrix([e_min_row,e_min_col])
    E_mat[e_min_row, e_min_col] = np.nan
    while(np.isnan(E_mat).all()==False):
        e_min_row, e_min_col = np.unravel_index(np.nanargmin(E_mat),E_mat.shape)
        if (E_mat[e_min_row, e_min_col] > tau):
            break
        E_temp = np.matrix([e_min_row,e_min_col])
        E_new = np.append(E_new,E_temp, axis=0)
        E_mat[e_min_row, e_min_col] = np.nan
    F_parent = np.arange(docs)
    F_rank = np.ones(docs)
    for i in range(0,len(E_new)):
        if (F_parent[E_new[i,0]] != F_parent[E_new[i,1]]):
            if (F_rank[F_parent[E_new[i,0]]] >= F_rank[F_parent[E_new[i,1]]]):
                F_rank[F_parent[E_new[i,0]]] += F_rank[F_parent[E_new[i,1]]]
                F_parent[F_parent == E_new[i,1]] = F_parent[E_new[i,0]]
            else: 
                F_rank[F_parent[E_new[i,1]]] += F_rank[F_parent[E_new[i,0]]]
                F_parent[F_parent == E_new[i,0]] = F_parent[E_new[i,1]]
    cluster_parents = np.unique(F_parent)
    num_of_clusters = len(cluster_parents)
    clusters = []
    for i in range(0,num_of_clusters):
        clusters.append((i,np.where(F_parent==cluster_parents[i])))
    return(clusters, F_parent)


def cluster_comp_fn(terms,cluster_list,sim_mat):
    num_of_clusters = len(cluster_list)
    cluster_comp = np.empty((terms,num_of_clusters)) * np.nan   
    for i in range(0,num_of_clusters):
        cluster_comp[:,i] = sim_mat[:,cluster_list[i][1][0][0]]
        if (len(cluster_list[i][1][0])>1):
            for j in range(1,len(cluster_list[i][1][0])):
                cluster_comp[:,i] += sim_mat[:,cluster_list[i][1][0][j]]   
    return(cluster_comp)
 

def cluster_sentiment_fn(sentiment_docs,cluster_list):
    num_of_clusters = len(cluster_list)
    cluster_sentiment = np.empty((num_of_clusters,2)) * np.nan   
    for i in range(0,num_of_clusters):
        if (len(cluster_list[i][1][0]) == 1):
            cluster_sentiment[i,0] = sentiment_docs[cluster_list[i][1][0][0],0]
            cluster_sentiment[i,1] = sentiment_docs[cluster_list[i][1][0][0],1]
        else:
            arous = 0
            vale = 0
            for j in range(1,len(cluster_list[i][1][0])):
                arous += sentiment_docs[cluster_list[i][1][0][0],0]
                vale += sentiment_docs[cluster_list[i][1][0][0],1]
            cluster_sentiment[i,0] = arous / len(cluster_list[i][1][0])
            cluster_sentiment[i,1] = vale / len(cluster_list[i][1][0])
    return(cluster_sentiment)


# -----------------------------------------------------------------------
## Read in CSV

with open('C:/Users/csfield/MSA/Fall 2/Text Mining/TextMining.csv',mode = 'r', encoding = 'ascii', errors = 'ignore') as f:
    reader = csv.reader(f)
    your_list = list(reader)

doc = []
inmate_stuff = []
for i in range(1,len(your_list)):
    doc.append(your_list[i][6])
    inmate_stuff.append((your_list[i][0],your_list[i][1],your_list[i][2],your_list[i][3],your_list[i][7]))

# -----------------------------------------------------------------------
## NLTK Term Vectors

# Remove punctuation, then tokenize documents
punc = re.compile('[%s]' % re.escape(string.punctuation))
term_vec = [ ] 

for d in doc:
    d = d.lower()
    d = punc.sub('', d)
    term_vec.append(nltk.word_tokenize(d))


# -----------------------------------------------------------------------
## NLTK Stop Words
    
# Remove stop words from term vectors
stop_words = nltk.corpus.stopwords.words( 'english' )
stop_words_2 = ['im','yall','would','like','go','dont','know','want','that','one','take','let','get','ill','come','thats','thatll','tell','say','ye','yes']
stop_words_2 += ['go','let','got','ive','mr','ok','didnt','thing','man','make','us','row']

for i in range( 0, len( term_vec ) ):
    term_list = [ ]

    for term in term_vec[ i ]:
        if (term not in stop_words and term not in stop_words_2):
            term_list.append( term )
    
    term_vec[ i ] = term_list


# -----------------------------------------------------------------------
## NLTK Porter Stemming

# Porter stem remaining terms
porter = nltk.stem.porter.PorterStemmer()

for i in range( 0, len( term_vec ) ):
    for j in range( 0, len( term_vec[ i ] ) ):
        term_vec[ i ][ j ] = porter.stem( term_vec[ i ][ j ] )

'''
# Print term vectors with stop words removed
for vec in term_vec:
    print(vec)
'''
     
# -----------------------------------------------------------------------
## Manually Calculate TF-IDF

n = len(doc)
unique_tv = list(set(x for l in term_vec for x in l))

tv_tf,tv_idf,tv_w,tv_w_norm = TF_IDF_fn(term_vec,unique_tv)
tv_tfidf = tv_w/tv_w_norm

unique_tv = unique_term_shrink_fn(tv_idf,unique_tv, z,n)
m = len(unique_tv)

tv_tf,tv_idf,tv_w,tv_w_norm = TF_IDF_fn(term_vec,unique_tv)
tv_tfidf = np.zeros((m,n))

while (len(np.where(tv_w_norm == 0)[0]) != 0):
    for zero_docs in reversed(np.where(tv_w_norm == 0)[0]):
        del term_vec[zero_docs]
        del doc[zero_docs]
        del inmate_stuff[zero_docs]
    
    n = len(term_vec)
    
    tv_tf,tv_idf,tv_w,tv_w_norm = TF_IDF_fn(term_vec,unique_tv)
    tv_tfidf = tv_w/tv_w_norm
    
    unique_tv = unique_term_shrink_fn(tv_idf,unique_tv, z,n)
    tv_tf,tv_idf,tv_w,tv_w_norm = TF_IDF_fn(term_vec,unique_tv)

tv_tfidf = tv_w/tv_w_norm


# -----------------------------------------------------------------------
## Latent Semantic Analysis

s,X_k = LSA_mat_fn(tv_tfidf, k)


# -----------------------------------------------------------------------
## Document Comparisons

m,n = np.shape(tv_tfidf)

doc_compare = np.zeros((n,n))
doc_compare_2 = np.zeros((n,n))

for j in range(0,n):
    for j2 in range(0,n):
        doc_compare[j,j2] = np.dot(tv_tfidf[:,j],tv_tfidf[:,j2])
        doc_compare_2[j,j2] = np.dot(X_k[:,j],X_k[:,j2])

# -----------------------------------------------------------------------
## Clustering
# Minimum Spanning Tree Clustering

clusters_TFIDF, parents_TFIDF = MST_tau_fn(doc_compare, tau)
clusters_LSA, parents_LSA = MST_tau_fn(doc_compare_2, tau)

cluster_tf_TFIDF = norm_mat_fn(tv_tf,0)

cluster_comp_TFIDF = cluster_comp_fn(m,clusters_TFIDF,cluster_tf_TFIDF)
cluster_comp_LSA = cluster_comp_fn(m,clusters_LSA,X_k)

cluster_words_TFIDF = max_words_fn(unique_tv, cluster_comp_TFIDF, max_words)
cluster_words_LSA = max_words_fn(unique_tv, cluster_comp_LSA, max_words)

print(len(clusters_TFIDF))
print(len(clusters_LSA))

cluster_size_TFIDF = np.zeros(len(clusters_TFIDF))
for i in range(0,len(clusters_TFIDF)):
    cluster_size_TFIDF[i] = len(clusters_TFIDF[i][1][0])

cluster_size_LSA = np.zeros(len(clusters_LSA))
for i in range(0,len(clusters_LSA)):
    cluster_size_LSA[i] = len(clusters_LSA[i][1][0])


# Popular Words in Each Cluster



# -----------------------------------------------------------------------
## NLTK Term Vectors

# Remove punctuation, then tokenize documents
punc = re.compile('[%s]' % re.escape(string.punctuation))
term_vec_NEW = [ ] 

for d in doc:
    d = d.lower()
    d = punc.sub('', d)
    term_vec_NEW.append(nltk.word_tokenize(d))

# -----------------------------------------------------------------------
## NLTK Stop Words
    
# Remove stop words from term vectors
stop_words = nltk.corpus.stopwords.words( 'english' )
stop_words_2 = ['im','yall','would','like','go','dont','know','want','that','one','take','let','get','ill','come','thats','thatll','tell','say','ye','yes']
stop_words_2 += ['go','let','got','ive','mr','ok','didnt','thing','man','make','us','row']

for i in range( 0, len( term_vec_NEW ) ):
    term_list = [ ]

    for term in term_vec_NEW[ i ]:
        if (term not in stop_words and term not in stop_words_2):
            term_list.append( term )
    
    term_vec_NEW[ i ] = term_list


# -----------------------------------------------------------------------
## Term Sentiment

temp_sent = []
sentiments = np.zeros((len(term_vec_NEW),2))
for i in range(0,len(term_vec_NEW)):
    temp_sent = []
    temp_sent = str(sentiment.sentiment(term_vec_NEW[i]))
    temp_sent = temp_sent.replace('{','')
    temp_sent = temp_sent.replace('}','')
    temp_sent = temp_sent.replace("'",'')
    temp_sent = temp_sent.split(sep=',')
    temp_sent[0] = temp_sent[0].split(sep=':')
    temp_sent[1] = temp_sent[1].split(sep=':')
    
    sentiments[i,0] = temp_sent[0][1]
    sentiments[i,1] = temp_sent[1][1]

print(np.mean(sentiments[:,0]))
print(np.mean(sentiments[:,1]))

cluster_sentiment = cluster_sentiment_fn(sentiments,clusters_LSA)


'''    
# Apply sentiment analysis 
sid = SentimentIntensityAnalyzer()

sid.polarity_scores(doc[400])

sid.polarity_scores(doc[0])

sentiment_analysis=[]
for i in doc[0:2]:
    #print(i)
    sa = sid.polarity_scores(i)
    sentiment_analysis.append(sa)
    for k in sorted(sa):
        print('{0}: {1}, '.format(k, sa[k]), end='')
    print("\n")
'''


df_inmates = pd.DataFrame(inmate_stuff)
df_inmates.columns = ["employee_num", "age", "county", "date_executed", "race"]
df_inmates["sent_arousal"] = sentiments[:,0]
df_inmates["sent_valence"] = sentiments[:,1]
df_inmates["cluster_parent"] = parents_LSA

df_inmates.to_csv("inmates_try1.csv")

df_clusters = pd.DataFrame(cluster_size_LSA)
df_clusters.columns = ["size"]
df_clusters["words"] = cluster_words_LSA
df_clusters["sent_arousal"] = cluster_sentiment[:,0]
df_clusters["sent_valence"] = cluster_sentiment[:,1]

df_clusters.to_csv("clusters.csv")
