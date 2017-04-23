# raw text-> removal of stop words, tokenize and stem using nltk Snowball
# created tf-idf matrix using scikit. Feature names are extracted and used as labels in dendrogram
# created similarity matrix using cosine similarity
# k-means clustering for documents and words. Here we start with a k=2 value and increase it in every iteration
# Use the cosine distance between data points and cetroid to decide on the k value
#The minute the value becomes same we stop the iteration with different k values
#To cross check we can see that the _inertia value reaches the elbow point generally used to determine the k value
#dumping the document similarity and word similarity matrix into a json file for chord diagram
from __future__ import print_function
from collections import Counter
from nltk.corpus import stopwords
import re
import codecs
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from scipy.cluster.hierarchy import ward, dendrogram,linkage,fcluster,cophenet,distance
import scipy.cluster.hierarchy as hier
import matplotlib.pyplot as plt
import json
import random
import numpy as np
from collections import defaultdict
import sys
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from decimal import *
#creating a set of stopwords provided by nltk
stopword=set(stopwords.words("english"))
#instatiating the class SnowballStemmer for stemming and getting root words
stemmer=SnowballStemmer("english")

def get_cluster_classes(den, label='ivl'):
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))

    cluster_classes ={}
    for c, l in cluster_idxs.items():
        i_l = [den[label][i] for i in l]
        cluster_classes[c] = i_l

    return cluster_classes

def tokenize(raw_text):
    tokens=re.findall('[a-zA-Z]+',raw_text.lower())
    return set(tokens)-stopword

def tokenize_and_stem(raw_text):
    tokens=re.findall('[a-zA-Z]+',raw_text.lower())
    allwords_tokenize=set(tokens) - stopword
    return [stemmer.stem(t) for t in allwords_tokenize if len(t)>2]

def get_files(raw_path):
    files_extracted=[]
    root_extracted=[]
    print ("Files in: " +raw_path)
    root_extracted.append(raw_path)
    for current_doc_id, current_file in enumerate(os.listdir(raw_path)):
        files_extracted.append(current_file)
    return files_extracted,root_extracted



def linkage_matrix_rep(sim_matrix):
    methods=['average','single','complete','weighted']
    c_final=0.0
    method_final=''
    final_linkage=linkage(sim_matrix)
    for method in methods:
        linkage_matrix = linkage(sim_matrix,method=method)
        c, coph_dists = cophenet(linkage_matrix, distance.pdist(sim_matrix))
        if c>c_final:
            c_final=c
            final_linkage=linkage_matrix
            method_final=method
            cd_final=coph_dists
    return c_final,method_final,final_linkage,cd_final

def file_extract(roots,files):
    genre_doc=[]
    for root in roots:
        for filename in files:
            with codecs.open(root+'\\'+filename, "r",encoding='utf-8', errors='ignore') as file_name:
                text=file_name.read()
                genre_doc.append(text)
    return genre_doc

def main(args):
    #input directory
    dir_to_process=args
    files,roots=get_files(dir_to_process)


    totalvocab_tokenized=[]
    totalvocab_stemmed=[]

    ebook=""
    ebooks=[]
    doc_name=[]


    #tokenization,removal of stopwords,stemming
    for root in roots:
        for filename in files:
            with codecs.open(root+'\\'+filename, "r",encoding='utf-8', errors='ignore') as file_name:
                text=file_name.read()

                ebook=ebook+"\n"+text
                ebooks.append(text)

                doc_name.append(filename)
                allwords_tokenize=tokenize(text)

                totalvocab_stemmed.extend([stemmer.stem(t) for t in allwords_tokenize])
                totalvocab_tokenized.extend(allwords_tokenize)
            file_name.close()

    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
    print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
    print (vocab_frame.head())

    #Creation of tf-idf matrix and vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=200,min_df=0.01,
                                      stop_words='english',
                                     tokenizer=tokenize_and_stem,ngram_range=(1,3),dtype='double')
    tfidf_matrix= tfidf_vectorizer.fit_transform(ebooks) #fit the vectorizer to synopses
    terms = tfidf_vectorizer.get_feature_names()

    #cosine distance for documents
    doc_sim =1-cosine_similarity(tfidf_matrix)

    # clustering using hierarchical clustering for documents
    #doc_cophen:cophenectic correlation value,doc_method:method used for calculating cophenetic distance,doc_linkage_matrix:linkage matrix,doc_cd:cophenetic distance
    doc_cophen,doc_method,doc_linkage_matrix,doc_cd = linkage_matrix_rep(doc_sim)

    #k-means clustering:document-document clustering
    num_clusters = 2
    getcontext().prec=2
    avg_distance=0.0
    flag=True
    prev_iter=0.0
    current_iter=0.0
    random.seed(10)
    while flag:
        km = KMeans(n_clusters=num_clusters,n_init= 1)
        km.fit(tfidf_matrix)
        joblib.dump(km,'doc_cluster.pkl')
        #km = joblib.load('doc_cluster.pkl')
        clusters = km.labels_.tolist()
        print(km.inertia_)
        centers=km.cluster_centers_
        sum_dist=0.0
        for i in range(0,len(doc_name)):
            clus=clusters[i]
            center=centers[clus:]
            doc=tfidf_matrix[i:]
            dist=1-cosine_similarity(doc,center)
            sum_dist=sum_dist+dist[0][0]
        avg=Decimal(sum_dist)/Decimal(len(doc_name))
        #print (sum_dist)
        print(avg)
        current_iter=Decimal(avg)-Decimal(avg_distance)
        if Decimal(prev_iter)-Decimal(current_iter)==Decimal(0):
            flag=False
        else:
            prev_iter=current_iter
            num_clusters=num_clusters+1

    print(clusters)
    cluster_doc=pd.DataFrame({"doc_cluster":clusters,"doc_name":doc_name})
    cluster_doc.to_csv('doc_to_cluster_map.csv',sep=',',index=False)

    #formation of dendrogram for document-document similarity
    fig, ax = plt.subplots(figsize=(15,20)) # set size
    ax = dendrogram(doc_linkage_matrix, orientation="left", labels=doc_name)
    plt.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')

    plt.tight_layout() #show plot with tight layout

    #save figure of the document clustering dendrogram
    plt.savefig('cosine_cluster_doc_test.png', dpi=200) #save figure as ward_clusters

    #getting the cluster to which document belongs
    doc_classes=get_cluster_classes(ax)
    thresh_doc=len(doc_classes)
    print(thresh_doc)

    #creating csv file containing document cluster,doc_id,doc_name
    cluster_index={}
    i=0
    doc_id=[]
    book_name=[]
    for c in doc_classes.keys():
        cluster_index[c]=i
        i=i+1
    for c in doc_classes.keys():
        for files in doc_classes[c]:
            doc_id.append(cluster_index[c])
            book_name.append(files)

    #Starting word clustering
    #word to word similarity
    word_vector=tfidf_matrix.transpose()
    word_vector=word_vector.A

    word_sim=1-cosine_similarity(word_vector)
    #print (word_sim)

    #linkage matrix created for the words
    word_cophen,word_method,word_linkage_matrix,word_cd = linkage_matrix_rep(word_sim)
    #print (word_cophen)
    #print (word_method)
    fig, ax = plt.subplots(figsize=(15, 20))
    final_terms=[]
    for term in terms:
        final_terms.append(vocab_frame.ix[term].values.tolist()[0][0])

    ax = dendrogram(word_linkage_matrix, orientation="left",labels=final_terms,show_contracted=True)
    plt.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')

    plt.tight_layout() #show plot with tight layout
    #saving figure of word-word clustering
    plt.savefig('cosine_cluster_word_test.png', dpi=200) #save figure as ward_clusters

    #Constructing a chord diagram
    r = lambda: random.randint(0,255)
    color=[]
    for i in range(len(doc_name)):
        color.append('#%02X%02X%02X' % (r(),r(),r()))

    #chord diagram:document-document similarity
    #color of arcs of document-document chord diagram
    doc_color=pd.DataFrame({'doc':doc_name,'color':color})
    doc_color.to_csv('C:\Users\Aishwarya Sadasivan\Python_Codes\js\dataset_doc.csv',sep=',',index=False)
    #creating json matrix of the document similarity matrix for percentage similarity chord diagram
    doc_sim_list=cosine_similarity(tfidf_matrix).tolist()
    #print(doc_sim_list)
    with open('C:\Users\Aishwarya Sadasivan\Python_Codes\js\doc_cos_dist.json', 'r') as f:
        json_data = json.load(f)
        json_data= doc_sim_list
    with open('C:\Users\Aishwarya Sadasivan\Python_Codes\js\doc_cos_dist.json', 'w') as f:
        f.write(json.dumps(json_data))


    #chord diagram:word-word similarity
    #colors of the arcs of chord diagram
    color_word=[]
    for i in range(len(terms)):
        color_word.append('#%02X%02X%02X' % (r(),r(),r()))
    word_color=pd.DataFrame({'word':terms,'color':color_word})
    word_color.to_csv('C:\Users\Aishwarya Sadasivan\Python_Codes\js\dataset-1_word.csv',sep=',',index=False)

    #purpose of displaying percentage of similarity in chord diagram
    word_sim_list=cosine_similarity(word_vector).tolist()
    #print(word_sim_list)

    with open('C:\Users\Aishwarya Sadasivan\Python_Codes\js\word_cos_dist.json', 'wb') as outfile:
        json.dump(word_sim_list, outfile)
        outfile.close()


if __name__=='__main__':
    main(sys.argv[1])
