import os
import json
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# nltk.download('all')
ps = PorterStemmer
from collections import OrderedDict
import matplotlib.pyplot as plot
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


################### INDEXING ##################################################

_WORD_MIN_LENGTH = 3
punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
_STOP_WORDS = stopwords.words('english') + list(punctuation) + ['\n']

#splitting the text into list of words and also store their locations in another list


def word_split(text):
    word_list = []
    wcurrent = []
    windex = None

    for i, c in enumerate(text):
        if c.isalnum():
            wcurrent.append(c)
            windex = i
        elif wcurrent:
            word = u''.join(wcurrent)
            word_list.append((windex - len(word) + 1, word))
            wcurrent = []

    if wcurrent:
        word = u''.join(wcurrent)
        word_list.append((windex - len(word) + 1, word))

    return word_list


#removing stop words and words of length less than the minimum length(here min length=3)

def words_cleanup(words):
    cleaned_words = []
    for index, word in words:
        if len(word) < _WORD_MIN_LENGTH or word in _STOP_WORDS:
            continue
        cleaned_words.append((index, word))
        # print(cleaned_words)
    return cleaned_words


#We normalize the words by converting them to lower case.

def words_normalize(words):
    normalized_words = []
    for index, word in words:
        wnormalized = word.lower()
        normalized_words.append((index, wnormalized))
    return normalized_words


#obtaining word tokens from text.

def word_index(text):
    words = word_split(text)
    words = words_normalize(words)
    words = words_cleanup(words)
    return words

#creating positional index for a document in the collection. Here, we store the locations of the terms in a document. 

def inverted_index(text):
    inverted = {}

    for index, word in word_index(text):
        locations = inverted.setdefault(word, [])
        locations.append(index)

    return inverted

#create positional index for the collection of documents. Here, We store doc ids and locations of the terms in each doc in a dictionary.

def inverted_index_add(inverted, doc_id, doc_index):
    for word, locations in doc_index.items():
        indices = inverted.setdefault(word, {})
        indices[doc_id] = locations
    return inverted

def req_inv_index(inverted):
    for doc_id, items in inverted.items():
        for k in items:
            items[k] = len(items[k])
    
    req_inverted={}
    for key , value in inverted.items():
        req_inverted[key]={}
        length = len(list(value))
        req_inverted[key][length]=value
        
    return req_inverted



################ SEARCHING #######################
# in the below function the idf values are stored in key via dictonary and i is the required item retrived from documents
#and calculating the idf values for required items.

def idf_values(req_inverted,docs_length):
    idf= {}
    for key,value in req_inverted.items():
        for i in value.keys():
            idf[key] = np.log2((docs_length) / i)
    return idf 

#we calculate tf-idf values for each document term pair
#from this function the we will ahve the complete tf-idf value from required_inverted and idf values.

def tf_idf_values(req_inverted,idf):
    tf_idf={}

    for key,value in req_inverted.items():
        tf_idf[key]={}
        for i,j in value.items():
            for x,y in j.items():
                tf_idf[key][x]= idf[key]*y
                # print(tf_idf[key])
    return tf_idf 

# in this fuction we are assigning the flags as the word index to each word "i - flag:  india 8"
def word_vec(idf):
    word_vector = {}
    flag=0
    # cnt = 0
    for i in idf.keys():
        word_vector[i] = flag;
        # if cnt < 10: 
        #     print("\ni - flag: ", i, flag)
        #     cnt+=1
        flag = flag+1
        # print(word_vector)
    return word_vector

#preprocessing the query and create the query vector. 

def query_vec(query,idf):
    query_vocab_stripped = list(set([x.strip('-.,!#$%^&*();:\n\t\\\"?!{}[]<>') for x in query.lower().split()]))
    query_list = [x.strip('-.,!#$%^&*();:\n\t\\\"?!{}[]<>') for x in query.lower().split()]
    query_wc = {}
    for word in query_vocab_stripped:
        query_wc[word] = query_list.count(word) #when the query is given if there is 3 words in the query first it will count no.of words and it will give for every word of query value as 1
        # print(query_wc[word])

    for key,value in query_wc.items():
        if key in idf.keys():
            query_wc[key] = value*idf[key]
            # print(query_wc[key])

    query_vector=np.zeros(len(idf))
    for i,j in idf.items():
        if i in query_wc.keys():
            query_vector[word_vector[i]]=query_wc[i]  
            # print(query_vector)
    return query_vector

def doc_vec(documents,idf,tf_idf):
    
    doc_vector={}
    for key,value in documents.items():
        doc_vector[key] = np.zeros(len(idf)) #number of items in an objects(len(idf)) and np.Zeros() will return the new array of given shape of elements of 0's
    for key,value in tf_idf.items():
        for j in documents.keys():
            if j in tf_idf[key].keys():
                doc_vector[j][word_vector[key]]= tf_idf[key][j]
     
    return doc_vector

#calculating cosine similarity between query and documents.

def cos_sim(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if (norm_a == 0) or (norm_b == 0):
        sim = 0
    else:
        sim=np.dot(a, b) / ( norm_a * norm_b )
    return sim 

#storing all cosine similarity values between query and documents and then we return relevance documents.

def mat_score(doc_vector,query_vector,query,k_value):
    f = open("relevance_feedback.json", 'r')
    relevance_feedback = json.load(f)
    f.close()
    relevant_queries = list(relevance_feedback.keys())
    print(relevant_queries)
    relevant_score = 0.1
    matching_score={}
    for key,value in doc_vector.items():
        vec = np.array(value)
        if query in relevant_queries and key in relevance_feedback[query]:
            matching_score[key] = cos_sim(query_vector,vec) +  relevant_score
        else: 
            matching_score[key] = cos_sim(query_vector,vec) 
    
    sorted_value = OrderedDict(sorted(matching_score.items(), key=lambda x: x[1], reverse = True))
    bestk = {k: [iter+1, sorted_value[k]] for iter, k in enumerate(list(sorted_value)[:k_value])}
    print(bestk)
    return bestk,matching_score


#################### EVALUATION ####################

#we create list of relevant documents
#beast k means here we will pesuod relevance docunments
def relevant_docs(bestk):
    relevancedocs=[]
    for i in bestk:
        relevancedocs.append(i)
    return relevancedocs


#we create list of matching scores
def mat_score_list(matching_score):
    ranking_query=[]
    for x in matching_score:
        temp=[]
        temp.append(x)
        temp.append(matching_score[x])
        ranking_query.append(temp)
    return ranking_query

def precision_rec(top,rel):
    relevant_retrieved=0
    retrieved=0
    precision=[]
    recall=[]
    print(len(top),len(rel))
    for i in top:
        retrieved=retrieved+1
        for j in rel:
            if(i==j):
             relevant_retrieved=relevant_retrieved+1
        print("precision = ",relevant_retrieved/retrieved,"recall = ",relevant_retrieved/len(rel))
        precision.append(relevant_retrieved/retrieved)
        recall.append(relevant_retrieved/len(rel))
    return precision,recall


# ##################### CAPTURING FEEDBACK #################

#we take user feedback,we obtain relevant documents vectors,non relevant documents vectors ,and then we calculate mean of
#relevant and non-relevant docs vectors,we add them to query vector with necessary percentage,Again we find cosine 
#similarity to the modified query vector.

def relevance_feedback(doc_vector,ranking_query,query_vector,query,k_value):
    
    total_query_vectors = []
    total_query_vectors.append(query_vector)
    rel_feedback=[]

    while(1):
        print("\n")
        rel_fedd = str(input("Enter space separated relevant documents indices(-1 to exit): "))
        print("..................relavance feedback....................")
        rel_fedd = rel_fedd.split()
        if(rel_fedd[0]!='-1'):
            rel_feedback=rel_fedd
    
        if int(rel_fedd[0])!=-1:
        
            rel_vectors = []
            non_rel_vectors = []

            for i in range(len(rel_fedd)):
                # ind_val = int(rel_fedd[i])
                tvect = doc_vector[rel_fedd[i]]
                rel_vectors.append(tvect)

            for i in range(len(bestk)):
                if str(i) not in rel_fedd:
                    tvect = doc_vector[ranking_query[i][0]]
                    non_rel_vectors.append(tvect)

            mean_rel_vector = np.zeros(len(rel_vectors[0]))
        
            for i in range(len(rel_vectors)):
                tv = np.array(rel_vectors[i])
                mean_rel_vector+=tv
            mean_rel_vector = mean_rel_vector/len(rel_vectors)

            mean_non_rel_vector = np.zeros(len(non_rel_vectors[0]))
        
            for i in range(len(non_rel_vectors)):
                tv = np.array(non_rel_vectors[i])
                mean_non_rel_vector+=tv
            mean_non_rel_vector = mean_non_rel_vector/len(non_rel_vectors)

            query_vector = (0.1 * query_vector) + (0.75 * mean_rel_vector) - (0.25 * mean_non_rel_vector)
            
            query_vector = np.array(query_vector)
            total_query_vectors.append(query_vector)
            
            top_k,matching_score = mat_score(doc_vector,query_vector,query,k_value)                
        else:
            break
    return rel_feedback,top_k,matching_score

####################  CALLINF FUNCTION #########################


# directory = 'players'
# directory = 'Dataset'
# directory = 'NewDataset'
directory = 'articles'

documents = {}
inverted = {}

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        infile = open(f, 'r', encoding="utf-8")
        words = infile.readlines()
        new = os.path.splitext(filename)[0]
        filename = new
        total_file_content = ""
        for word in words:
            total_file_content += word
        # documents[filename] = words[0]
        documents[filename] = total_file_content

docs_length = len(documents) 

for doc_id, text in documents.items():
    doc_idx = inverted_index(text)
    inverted_index_add(inverted, doc_id, doc_idx)
    
    

req_inverted = req_inv_index(inverted)
idf = idf_values(req_inverted,docs_length)
tf_idf = tf_idf_values(req_inverted,idf)
word_vector = word_vec(idf)

query = str(input("Enter the query to retrieve documents: "))
query_vector = query_vec(query,idf)
doc_vector = doc_vec(documents,idf,tf_idf)
doctlist = list(doc_vector.items())
# print(doctlist[0])
k_value = int(input("Enter the number of documents to be retrieved for the query: "))
print("\n")
bestk,matching_score = mat_score(doc_vector,query_vector,query,k_value)
relevancedocs = relevant_docs(bestk)
ranking_query = mat_score_list(matching_score)


rel_feedback_array,best_k,matching_score = relevance_feedback(doc_vector,ranking_query,query_vector,query,k_value)

relevance_docs = relevant_docs(best_k)
rank_query = mat_score_list(matching_score)
print("\n")
print("PR-Curve for given query")
p,r=precision_rec(bestk,rel_feedback_array)
plot.plot(r, p)
  
# naming the x axis
plot.xlabel('recall')
# naming the y axis
plot.ylabel('precision')
  
# giving a title to my graph
plot.title('pr-curve')
  
# function to show the plot
plot.show()
# plot.show()

import json
item = dict()
item[query] = rel_feedback_array
print(item)
with open("relevance_feedback.json", "w", encoding='utf-8') as file:
    json.dump(item, file, indent=4)
    file.close()