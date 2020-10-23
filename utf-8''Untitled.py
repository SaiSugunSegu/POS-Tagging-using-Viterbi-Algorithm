#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import defaultdict
import string
import math


# In[2]:


with open('WSJ_training.pos') as f:    # Training data from Wall Street Journal
    training_corpus = f.readlines()    

    print("Training corpus list:", len(training_corpus))
print(training_corpus[:5])


# In[3]:


# Preparing vocabulary
# 1) Taking the words out from training set
words = [line.split('\t')[0] for line in training_corpus]
print('words',words[:7])

# 2) Creating default dictnary of word_count
word_count = defaultdict(int)
for i in words:
    word_count[i] += 1

# 3) vocab = words whos freqency is more than 2
vocab_l = [k for k, v in word_count.items() if (v > 1 and k != '\n')]
print('vocab',vocab_l[:7])

# 4) sort and give unique number 
vocab = {}
for i, word in enumerate(vocab_l): 
    vocab[word] = i 


# In[4]:


# Assign tags to unknown words
def assign_unk(word):
    punct = set(string.punctuation)
    
    # Suffixes
    noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
    verb_suffix = ["ate", "ify", "ise", "ize"]
    adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
    adv_suffix = ["ward", "wards", "wise"]

    # digit
    if any(char.isdigit() for char in word):
        return "--unk_digit--"
    # punctuation character
    elif any(char in punct for char in word):
        return "--unk_punct--"
    # upper case character
    elif any(char.isupper() for char in word):
        return "--unk_upper--"
    # noun suffix
    elif any(word.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"
    # verb suffix
    elif any(word.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"
    # adjective suffix
    elif any(word.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"
    # adverb suffix
    elif any(word.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"  
    # If none of the previous criteria is met, return plain unknown
    return "--unk--"


# In[5]:


def get_word_tag(line, vocab):
    if not line.split():             # If line is empty return placeholders for word and tag
        word = "--n--"
        tag = "--s--"
    else:
        word, tag = line.split()
        if word not in vocab: 
            word = assign_unk(word)
    return word, tag


# In[6]:


def create_dictionaries(training_corpus, vocab):
    transition_counts = defaultdict(int)
    emission_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    
    prev_tag = '--s--' 
    for word_tag in training_corpus:
        word, tag = get_word_tag(word_tag,vocab)
        transition_counts[(prev_tag, tag)] += 1
        emission_counts[(tag,word)]+= 1
        tag_counts[tag] += 1
        prev_tag = tag
    return transition_counts, emission_counts, tag_counts


# In[7]:


transition_counts, emission_counts, tag_counts = create_dictionaries(training_corpus, vocab)


# In[8]:


print(len(transition_counts))
print("transition_count examples: ")
for ex in list(transition_counts.items())[:7]:
    print(ex)
print()
print(len(emission_counts))
print("emission_count examples: ")
for ex in list(emission_counts.items())[:7]:
    print (ex)
print()  
print(len(tag_counts))
print("tag_count examples: ")
for ex in list(tag_counts.items())[:7]:
    print (ex)


# In[9]:


def create_transition_matrix(alpha, transition_counts, tag_counts):
    all_tags = list(tag_counts.keys())
    print(all_tags)
    num_tags = len(all_tags)
    
    A = np.zeros((num_tags,num_tags))
    
    trans_keys = set(list(transition_counts.keys()))

    for pre_tag in range(num_tags): 
        for tag in range(num_tags):
            count = 0
            key = (all_tags[pre_tag],all_tags[tag])

            if key in trans_keys: 
                count = transition_counts[key]
                
            prev_tag_count = tag_counts[all_tags[pre_tag]]
            
            A[pre_tag, tag] = (count + alpha) / (prev_tag_count + alpha * num_tags)
            
    transition_matrix = pd.DataFrame(A, index=all_tags, columns = all_tags)
    return transition_matrix


# In[10]:


alpha = 0.01
transition_matrix = create_transition_matrix(alpha, transition_counts, tag_counts)
transition_matrix.iloc[:5, :]


# In[11]:


def create_emission_matrix(alpha, emission_counts, tag_counts, vocab):
    all_tags = list(tag_counts.keys())
    
    num_tags = len(all_tags)
    num_words = len(vocab)
    
    B = np.zeros((num_tags,num_words))
    
    emis_keys = set(list(emission_counts.keys()))

    for tag in range(num_tags): 
        for word in range(num_words):
            count = 0
            key = (all_tags[tag],vocab[word])

            if key in emis_keys: 
                count = emission_counts[key]
                
            tag_count = tag_counts[all_tags[tag]]
            
            B[tag,word] = (count + alpha) / (tag_count + alpha * num_words)
            
    emission_matrix = pd.DataFrame(B, index=all_tags, columns = vocab)
    return emission_matrix


# In[12]:


alpha = 0.01
emission_matrix = create_emission_matrix(alpha, emission_counts, tag_counts, list(vocab))
emission_matrix.iloc[:5,:]


# In[13]:


# Test Corpus


# In[14]:


with open('WSJ_testing.pos') as f:    # Training data from Wall Street Journal
    testing_corpus = f.readlines()    

    print("Training corpus list:", len(testing_corpus))
print(testing_corpus[:5])
test_words = [line.split('\t')[0] for line in testing_corpus]


# In[15]:


def preprocess(vocab, test_corpus):
    prep = []
    # Read data
    for word in test_corpus:
        # End of sentence
        if not word.split():
            word = "--n--"
            prep.append(word)
        # Handle unknown words
        elif word.strip() not in vocab:
            word = assign_unk(word)
            prep.append(word)
        else:
            prep.append(word.strip())

    return prep


# In[16]:


corpus = preprocess(vocab, test_words) 


# In[17]:


print(test_words[:13])
print(corpus[:13])


# In[18]:


def initialize(tag_counts, A, B, corpus, vocab):
    num_tags = len(tag_counts)
    all_tags = list(tag_counts.keys())
    
    best_probs = np.zeros((num_tags, len(corpus)))
    best_probs = pd.DataFrame(best_probs, index=all_tags, columns = corpus)
    best_paths = np.zeros((num_tags, len(corpus)))
    best_paths = pd.DataFrame(best_paths, index=all_tags, columns = corpus)
    
    s_idx = "--s--"
    
    for tag in all_tags:
        if A.loc[s_idx,tag] == 0: 
            best_probs.loc[tag,corpus[0]] = float("-inf")
        else:
            best_probs.loc[tag,corpus[0]] = math.log(A.loc[s_idx,tag]) + math.log(B.loc[tag,corpus[0]])
                       
    return best_probs, best_paths


# In[19]:


best_probs, best_paths = initialize(tag_counts, transition_matrix, emission_matrix, corpus, vocab)


# In[20]:


best_probs.iloc[:7,:]


# In[23]:


def viterbi_forward(A, B,tag_counts, corpus, best_probs, best_paths, vocab):
    num_tags = len(tag_counts)
    all_tags = list(tag_counts.keys())
    
    for word in range(1,len(corpus[:200])): 
        for prev_tag in range(num_tags): 
            best_prob_word =  float("-inf")
            best_path_word = None
            for tag in range(num_tags):
                prob = best_probs.iloc[prev_tag,word-1] + math.log(A.iloc[prev_tag,tag]) + math.log(B.iloc[tag,word]) 
                if prob > best_prob_word: 
                    best_prob_word = prob
                    best_path_word = tag
            best_probs.iloc[prev_tag,word] = best_prob_word
            best_paths.iloc[prev_tag,word] = best_path_word
            
        ### END CODE HERE ###
    return best_probs, best_paths


# In[24]:


best_probs, best_paths = viterbi_forward(transition_matrix, emission_matrix, tag_counts, corpus, best_probs, best_paths, vocab)


# In[50]:


best_probs.iloc[:,:10]


# In[67]:


def viterbi_backward(best_probs, best_paths, corpus, tag_counts):
    all_tags = list(tag_counts.keys())
    num_tags = best_probs.shape[0]
    m = best_paths.shape[1]
    
    z = [None] * m
    best_prob_for_last_word = float('-inf')
    pred = [None] * m
    
    for k in range(num_tags):
        if best_probs.iloc[k,m-1] > best_prob_for_last_word: 
            best_prob_for_last_word = best_probs.iloc[k,m-1]
            z[m - 1] = k
            
    pred[m - 1] = all_tags[k]
    
    for i in range(m-1,-1,-1): 
        pos_tag_for_word_i = int(z[i])
        z[i - 1] = best_paths.iloc[pos_tag_for_word_i,i]
        pred[i - 1] = all_tags[int(z[i - 1])]
        
     ### END CODE HERE ###
    return pred


# In[72]:


pred = viterbi_backward(best_probs.iloc[:,:100], best_paths.iloc[:,:100], corpus, tag_counts)


# In[76]:


for i in range(100):
    print('Predicted POS for "{}" is "{}"'.format(corpus[i], pred[i]))


# In[ ]:




