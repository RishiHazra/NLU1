# -*- coding: utf-8 -*-
"""
@author: Rishi
###### IMPORT GUTENBERG CORPUS   #######


@@@-----------S2: Train: D2-Train, Test: D2-Test----------------@@@

"""
###### IMPORT GUTENBERG CORPUS   #######
import nltk
import re
import math
import string
import random
import numpy as np
nltk.download('gutenberg')
from nltk.corpus import gutenberg

categories=gutenberg.fileids()
gutenberg_corpus_sents={}
for cat in categories:
    a=[]
    for sents in gutenberg.sents(fileids=cat):
        sents.insert(0,'<s>')
        sents.append('<e>')
        a.append(sents)
    gutenberg_corpus_sents[cat]=a

GUTENBERG={} 
for cat in categories:
    a=[]
    for i in range(0,len(gutenberg_corpus_sents[cat])):
        for word in gutenberg_corpus_sents[cat][i]:
            a.append(word.lower())
    GUTENBERG[cat]=a    
    
"""
$$ splitting into train(80%) , dev set(10%) and test set(10%)  $$
"""

train=[]
test=[]
dev=[]
remaining=[]
whole_gutenberg=[]

for key,corpus in  GUTENBERG.items():
    split=math.floor(80*len(corpus)/100)
    train.append(corpus[:split])
    remaining.append(corpus[split:])
    whole_gutenberg+=GUTENBERG[key]

for corpus1 in remaining:
     split1=math.floor(len(corpus1)/2)
     test.append(corpus1[:split1])
     dev.append(corpus1[split1:])
   
whole_train=[]
whole_test=[]
whole_dev=[]
for i in range(len(categories)):
    whole_train+=train[i]
    whole_test+=test[i]  
    whole_dev+=dev[i]


# remove punctuation from each word

table = str.maketrans('', '','!"#$%&\'()*+,-/:;=?@[\\]^_`{|}~.')
new_train = [word.translate(table) for word in whole_train]
new_test=  [word.translate(table) for word in whole_test]
new_dev=   [word.translate(table) for word in whole_dev]
new_gutenberg= [word.translate(table) for word in whole_gutenberg]

new_train=list(filter(None,new_train))
new_test=list(filter(None,new_test))
new_dev=list(filter(None,new_dev))
new_gutenberg=list(filter(None,new_gutenberg))

''' Replace rare words in the corpus with '<UNK>''''

vocabulary={}

for word in new_train:
    if word in vocabulary:
        vocabulary[word]+=1
    else:
        vocabulary[word]=1
      
## contains all words with freq>1

vocabulary1={}
for word,value in vocabulary.items():
    if value>1:
        vocabulary1[word]=value
    else:
         a=np.random.choice(2,1, p=[0.2,0.8])
         if a[0]==1:
             vocabulary1[word]=value

new_train1=[]   
for word in new_train:
    if word in vocabulary1:
        new_train1.append(word)
    else:
        new_train1.append('<UNK>')
        
new_test1=[]
for word in new_test:
    if word in vocabulary1:
        new_test1.append(word)
    else:
        new_test1.append('<UNK>')
    
new_dev1=[]
for word in new_dev:
    if word in vocabulary1:
        new_dev1.append(word)
    else:
        new_dev1.append('<UNK>')   
        

"""
########  DESIGNING THE MODEL   ########
"""
    
import itertools
from itertools import tee, islice

def ngrams(lst, n):
  tlst = lst
  while True:
    a, b = tee(tlst)
    l = tuple(islice(a, n)) 
    if len(l) == n:
        yield l
        next(b)
        tlst = b
    else:
        break


### for train data ###
from collections import Counter
unigram = {}
for token in new_train1:
  if token not in unigram:
    unigram[token] = 1
  else:
    unigram[token] += 1

bigram=Counter(ngrams(new_train1,2))
del(bigram['<s>','<e>'])
trigram=Counter(ngrams(new_train1,3))
for keys in trigram.keys():
    if keys[0:2]==('<s>','<e>') or keys[1:3]==('<s>','<e>'):
        trigram[keys]=-1
trigram={keys:value for keys,value in trigram.items() if value!=-1}


### for whole gutenberg corpus ###
bigram3=Counter(ngrams(new_gutenberg,2))
del(bigram3['<s>','<e>'])
trigram3=Counter(ngrams(new_gutenberg,3))
for keys in trigram3.keys():
    if keys[0:2]==('<s>','<e>') or keys[1:3]==('<s>','<e>'):
        trigram3[keys]=-1
trigram3={keys:value for keys,value in trigram3.items() if value!=-1}

unigram3 = {}
for token in new_gutenberg:
  if token not in unigram3:
    unigram3[token] = 1
  else:
    unigram3[token] += 1

### for test set ###
unigram1 = {}
for token in new_test1:
  if token not in unigram1:
    unigram1[token] = 1
  else:
    unigram1[token] += 1


"""
###### KNESER NEY SMOOTHING  ######
"""

''' for bigram perplexity '''
unique_bigram={}

for keys in bigram.keys():
    if keys[1] in unique_bigram:
       unique_bigram[keys[1]]+=1
    else:
        unique_bigram[keys[1]]=1

unique_bigram_follow={}

for keys in bigram.keys():
    if keys[0] in unique_bigram_follow:
        unique_bigram_follow[keys[0]]+=1
    else:
        unique_bigram_follow[keys[0]]=1
        
d1=0.5
d2=0.75 
A={}
C={}
s=sum(unique_bigram.values())
for keys in bigram.keys():
    if keys[0] in A.keys():
        A[keys[0]][keys[1]]=max(bigram[keys]-d2,0)/unigram[keys[0]]
        b=(d2/unigram[keys[0]])*unique_bigram_follow[keys[0]]
        Pcont=unique_bigram[keys[1]]/s
        C[keys[1]]=b*Pcont
    else:
        A[keys[0]]={}
        A[keys[0]][keys[1]]=max(bigram[keys]-d2,0)/unigram[keys[0]]
        b=(d2/unigram[keys[0]])*unique_bigram_follow[keys[0]]
        Pcont=unique_bigram[keys[1]]/s
        C[keys[1]]=b*Pcont


'''' for trigram perplexity ''''
        
A_trigram={}
C_trigram={}
unique_trigram_follow={}
unique_trigram_precede={}

for keys in trigram.keys():
    if keys[2] in unique_trigram_precede:
       unique_trigram_precede[keys[2]]+=1
    else:
        unique_trigram_precede[keys[2]]=1

for keys in trigram.keys():
    if keys[0:2] in unique_trigram_follow:
        unique_trigram_follow[keys[0:2]]+=1
    else:
        unique_trigram_follow[keys[0:2]]=1

s=sum(unique_trigram_precede.values())

for keys in trigram.keys():
    if keys[0:2] in A_trigram:
        A_trigram[keys[0:2]][keys[2]]=max(trigram[keys]-d2,0)/bigram[keys[0:2]]
        b=(d2/bigram[keys[0:2]])*unique_trigram_follow[keys[0:2]]
        Pcont_trigram=unique_trigram_precede[keys[2]]/s
        C_trigram[keys[2]]=b*Pcont_trigram
    else:
        A_trigram[keys[0:2]]={}
        A_trigram[keys[0:2]][keys[2]]=max(trigram[keys]-d2,0)/bigram[keys[0:2]]
        b=(d2/bigram[keys[0:2]])*unique_trigram_follow[keys[0:2]]
        Pcont_trigram=unique_trigram_precede[keys[2]]/s
        C_trigram[keys[2]]=b*Pcont_trigram
            
            
"""
$$ CALCULATING PERPLEXITY OF EACH MODEL $$
"""

def perplexity(n):
    N=n
    prob=0
    
    if N==1:
       d=sum(unigram.values())
       for key in unigram1.keys():
           prob+=unigram1[key]*math.log(d/unigram[key],2)
       perplexity=pow(2, prob/sum(unigram1.values()))  
      
    if N==2:
        bigram1=Counter(ngrams(new_test1, 2))
        for keys in bigram1.keys():
            if bigram[keys]!=0:
               Prob=A[keys[0]][keys[1]]+C[keys[1]]
               prob+=bigram1[keys]*math.log(1/Prob,2)
            else:
                Prob=C[keys[1]]
                prob+=bigram1[keys]*math.log(1/Prob,2)
        perplexity=pow(2, prob/sum(bigram1.values())) 
       
    if N==3:
        trigram1=Counter(ngrams(new_test1,3))
        for keys in trigram1.keys():
            if keys in trigram:
                if trigram[keys]!=0:
                    Prob=A_trigram[keys[0:2]][keys[2]]+C_trigram[keys[2]]
                    prob+=trigram1[keys]*math.log(1/Prob,2)
                else:
                    Prob=C_trigram[keys[2]]
                    prob+=trigram1[keys]*math.log(1/Prob,2)
        perplexity=pow(2, prob/sum(trigram1.values())) 
  
    return perplexity    
        


"""
######## GENERATING SENTENCES  #########
"""

from collections import defaultdict
from random import choice

prob={}
for keys,value in trigram3.items():
    if keys[0:2] in prob:
       prob[keys[0:2]][keys[2]]=value/bigram3[keys[0:2]]
    else:
        prob[keys[0:2]]={}
        prob[keys[0:2]][keys[2]]=value/bigram3[keys[0:2]]
        
## sentence generator(random)
mapper = defaultdict(list)

for keys in trigram3:
    mapper[keys[0:2]].append(keys[2]) 
   
    
mapper1 = defaultdict(list)
for keys in bigram3:
    if bigram3[keys]!=0:
        mapper1[keys[0]].append(keys[1])
       


def generate_sentence(word1,N,p):
    
    l=p
    sentence=[word1]
    word2 = choice(mapper1[word1])
    while bigram3[word1,word2]/unigram3[word1] <0.02:
        word2 = choice(mapper1[word1])        
   
    sentence.append(word2)
    
    for i in range(l):
        new_word=choice(mapper[word1,word2])
        import random
        r=0.1*random.uniform(0,1)
        while prob[word1,word2][new_word]<r:
            new_word=choice(mapper[word1,word2])
            r=0.1*random.uniform(0,1)
       
        if new_word=='<e>':
            if prob[word1,word2]['<e>']==1:
                new_word=choice(mapper['<e>','<s>']) 
                word1, word2 = '<s>', new_word
            else:
                while new_word=='<e>':
                    new_word=choice(mapper[word1,word2])
                word1, word2 = word2, new_word
        else:
            word1,word2=word2,new_word
        sentence.append(new_word)
    word3,word4=word1,word2
    
    k=0;
         
    while word4!='<e>':
        word3,word4=word1,word2
        sentence3=[]
        for j in range(N-p):
            new_word1=choice(mapper[word3,word4])
            import random
            rand=0.1*random.uniform(0,1)
            while prob[word3,word4][new_word1]<rand:
                new_word=choice(mapper[word1,word2])
                rand=0.1*random.uniform(0,1)
                      
            if new_word1=='<e>':
                if j!=(N-p-1):
                    if prob[word3,word4]['<e>']==1:
                        new_word1=choice(mapper['<e>','<s>']) 
                        word3, word4 = '<s>', new_word1
                    else:
                        while new_word1=='<e>':
                            new_word1=choice(mapper[word3,word4])
                        word3, word4 = word4, new_word1
                else:
                    word4=new_word1
            else:
                word3,word4=word4,new_word1
            
            sentence3.append(new_word1) 
        k+=1
        if k>20:
            break

    if k<=20:
        one=" ".join(sentence)
        two=" ".join(sentence3)
        print(one,two)
    else:
        generate_sentence('<s>',N,p-1)
    

word1='<s>'
generate_sentence(word1,10,7)

print("perplexity_gutenberg for unigram= ",perplexity(1)) 
print("perplexity_gutenberg for bigram = ", perplexity(2))     
print("perplexity_gutenberg for trigram= ",perplexity(3))

"""#######-------------------------END-------------------##########"""