# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 06:16:46 2017

@author: Rajat Biswas
"""
import nltk
import math
import string
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords
import time 
import pickle
from collections import OrderedDict as OD 
import queue as Q
from heapq_max import *
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.properties import StringProperty,ObjectProperty
from kivy.base import runTouchApp
from kivy.lang import Builder
from kivy.uix.treeview import TreeView, TreeViewNode
from kivy.uix.treeview import TreeViewLabel
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from functools import partial 
from nltk.corpus import words
import re

word_list=[]#contains all the unstemmed tokens in the corpus

word_set=set()#Contains all the unique unstemmed words

stop_words=set(stopwords.words("english"))#Preparing a set of stopwords in english

ps=PorterStemmer()#Creating the stemmer

idss=nltk.corpus.movie_reviews.fileids()#File names of all the documents in the corpus 

docs=[]#Reading the raw documents in the corpus



translator=str.maketrans(' ',' ',string.punctuation)#MAkeing a translator table for passing to translate
for x in idss:
    str(movie_reviews.raw(x)).replace('-',' ')
    docs.append(movie_reviews.raw(x).translate(translator))#stripping  off the punctuations from the raw data
    


q_idf=OD()#dictionary used for chekcing with the query


magn_list=[]#List storing the magnitude of each document vector


k_best=[]#IT will store the final result


def preprocessing(raw_docs):
    tokenized_docs=[]
    dummy=[]#First storing the tokenised docs in this 
    for x in raw_docs:
        dummy=word_tokenize(x)
        temp=[]
        for y in dummy:
            
            if(y not in stop_words):#stripping off the stopwords from raw data
                temp.append(ps.stem(y))#adding stem words to dictionary
                word_set.add(y)
        tokenized_docs.append(temp)
    return tokenized_docs

#Creating a list for auto complete
preprocessing(docs)
for x in word_set:
    if(len(x)>=3):
        word_list.append(x)

word_list=sorted(word_list)

print(len(word_list))



 
def term_frequency(term,document):
    count=document.count(term)
    if(count==0):
        return 0
    return 1+math.log(count)
    
def inverse_document_frequency(tokenized_documents):
    idf=OD()
    all_tokens=set()
    for x in tokenized_documents:
        for word in x:
            all_tokens.add(word)
    #print("Size of docs"+str(len(tokenized_documents)))
    for token in all_tokens:
        word_list.append(token)
        counter=0
        for x in tokenized_documents:
            if (token in x):
                counter+=1
        idf[token]=1+math.log((len(tokenized_documents))/(counter))
    return idf
    
"""Special BLock"""    
    
#q_idf=inverse_document_frequency(preprocessing(docs))
#sorted(q_idf)
"""ENds here"""
    
def tf_idf(documents):
    tokenized_docs=preprocessing(documents)
    #q_idf=inverse_document_frequency(tokenized_docs)
    
    tfidf_documents=[]
    for document in tokenized_docs:
        doc_tfidf=[]
        for term in q_idf.keys():
            tf=term_frequency(term,document)
            doc_tfidf.append(tf*q_idf[term])
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents

t1=time.time()





#q_idf=inverse_document_frequency(preprocessing(docs))
#print("Created the inverse document frequency")
#pickle.dump(q_idf,open("idf.pickle","wb"))
print("Reading  idf from a pickle file ")
q_idf=pickle.load(open("idf.pickle","rb"))
#sorted(q_idf)

t2=time.time()
#tfidf_final=tf_idf(docs)

tfidf_final=pickle.load(open("tfidf.pickle","rb"))#Loading the tfidf
"""
Now Calculating the magnitudeVector for query optimisation

"""
for x in tfidf_final:
    sum=0
    for y in x:
        sum+=y*y
    sum=sum**(0.5)
    magn_list.append(sum)#Creating the list of magnitudes of vectors

print(t2-t1)    

#pickle.dump(tfidf_final,open("tfidf.pickle","wb"))
#print("wrote the vectorizer into pickle file")


print("read tfidf  from the pickle file")

t3=time.time()
print(t3-t2) 



def cosine_similarity(v1,v2,ind_list,mag,ind):
    dot=0
    for x in ind_list:
        dot+=v1[x]*v2[x]

    magnitude = mag*(magn_list[ind])
    if(magnitude==0):
        return 0;
    return dot/magnitude

def experience(l1):
    word_list=l1+word_list
     

Builder.load_file("scroll.kv")
#For suggestions and taking input
class MyTextInput(TextInput):
    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        if keycode[1] == 'spacebar':
            #print("hsyvguhs")
            self.suggestion_text = '    '
            return True
        if self.suggestion_text and keycode[1] == 'tab':
            self.insert_text(self.suggestion_text +' ')
            return True
        
        return super(MyTextInput, self).keyboard_on_key_down(window, keycode, text, modifiers)
    
    def on_text(self, instance, value):
        self.suggestion_text = '    '
        val = value[value.rfind(' ') + 1:]
        if not val:
            return
        try:
            
            word = [word for word in word_list
                    if word.startswith(val)][0][len(val):]
            if not word:
                #return 
                word1 = [word1 for word1 in word_list
                    if word1.startswith(val)][1][len(val):]
                if not word1:
                    self.suggestion_text='    '
                    return
                self.suggestion_text=word1
                return
            self.suggestion_text = word
        except IndexError:
            self.suggestion_text = '    '
            print ('Index Error.')
            
            

print(len(word_list))
class ScrollableLabel(ScrollView):
    ll1=StringProperty('')
    ll2=StringProperty('')
    ll3=StringProperty('')
    ll4=StringProperty('')
    ll5=StringProperty('')
    ll6=StringProperty('')
    ll7=StringProperty('')
    ll8=StringProperty('')
    ll9=StringProperty('')
    ll10=StringProperty('')
    name=[]
    score=[]
    l=[]#TO store the documents to be shown
    sts=[]#Store the states of the button
    
    def bPress1(self):
        if (self.sts[0] == 1):
            self.ll1=''
            self.sts[0]=0
        else:
            self.ll1=self.l[0]
            self.sts[0]=1
            
    def bPress2(self):
        if (self.sts[1] == 1):
            self.ll2=''
            self.sts[1]=0
        else:
            self.ll2=self.l[1]
            self.sts[1]=1
            
    def bPress3(self):
        if (self.sts[2] == 1):
            self.ll3=''
            self.sts[2]=0
        else:
            self.ll3=self.l[2]
            self.sts[2]=1
    
    def bPress4(self):
        if (self.sts[3] == 1):
            self.ll4=''
            self.sts[3]=0
        else:
            self.ll4=self.l[3]
            self.sts[3]=1
    
    def bPress5(self):
        if (self.sts[4] == 1):
            self.ll5=''
            self.sts[4]=0
        else:
            self.ll5=self.l[4]
            self.sts[4]=1
    
    def bPress6(self):
        if (self.sts[5] == 1):
            self.ll6=''
            self.sts[5]=0
        else:
            self.ll6=self.l[5]
            self.sts[5]=1
            
    def bPress7(self):
        if (self.sts[6] == 1):
            self.ll7=''
            self.sts[6]=0
        else:
            self.ll7=self.l[6]
            self.sts[6]=1
            
    def bPress8(self):
        if (self.sts[7] == 1):
            self.ll8=''
            self.sts[7]=0
        else:
            self.ll8=self.l[7]
            self.sts[7]=1
            
    def bPress9(self):
        if (self.sts[8] == 1):
            self.ll9=''
            self.sts[8]=0
        else:
            self.ll9=self.l[8]
            self.sts[8]=1
    
    def bPress10(self):
        if (self.sts[9] == 1):
            self.ll10=''
            self.sts[9]=0
        else:
            self.ll10=self.l[9]
            self.sts[9]=1
    
    def queryResult(self):
        t4=time.time()
        self.ll1=''
        self.ll2=''
        self.ll3=''
        self.ll4=''
        self.ll5=''
        self.ll6=''
        self.ll7=''
        self.ll8=''
        self.ll9=''
        self.ll10=''
        self.score.clear()
        self.name.clear()
        self.sts.clear()
        self.l.clear()
        self.ids['status'].text=""
        query=""
        query=self.ids['search'].text
        query_tfidf=[]
        dummy_query=word_tokenize(query)
        global word_list
        query_tokens=[]
        for x in dummy_query:
            word_list.insert(0,x)
            query_tokens.append(ps.stem(x))
        i=0
        ind_list=[]#Contains the list of indices with whom we have to do the dot product
        m_query=0#Storing the magnitude of query 
    
        for x in q_idf.keys():
            if(x in query_tokens):
                query_tfidf.append(q_idf[x])
                ind_list.append(i)
                m_query+=(q_idf[x])**2
            else:
                query_tfidf.append(0)
            i+=1
        m_query=m_query**(0.5)
    
    
        mx=-1  #cosine similarity with other docs
        idx=0#Storing the index of the doc for accessing the precomputed value in magn_list
    
        for ind,x in enumerate(tfidf_final):
            if(cosine_similarity(x,query_tfidf,ind_list,m_query,ind)>mx):
                mx=cosine_similarity(x,query_tfidf,ind_list,m_query,ind)
                idx=ind
            var=1.0
            topt=str(movie_reviews.raw(idss[ind]))
            for y in dummy_query:
                tl=len(re.findall(" "+y+" ",topt))
                if(tl>=1):
                    var=var*(2**tl)
                    
            k_best.append([var*(cosine_similarity(x,query_tfidf,ind_list,m_query,ind)),ind])
    
        heapify_max(k_best)
        val=[]#TO store the tuples after being popped 
        
        no_results=-1
        rev=""
        for x in range(10):
            temp=heappop_max(k_best)
            val.append(temp)
            if(temp[0]==0):#Checking for base condition
                no_results=x
                print("NO results found")
                
            self.ids[str('b'+str(x+1))].text=str('')
            
            rev="[b]"+str(movie_reviews.raw(idss[temp[1]]))+"[/b]"
            for s in dummy_query:
                rev=rev.replace(""+s+"","[color=ff3333]"+s+"[/color]")
                
            self.score.append(temp[0])
            self.name.append(idss[temp[1]])
            
            print(temp[0])
            print(idss[temp[1]])
            #self.labels[x].text=rev
            
            self.sts.append(0)
            if(no_results==-1):
                self.ids[str('b'+str(x+1))].height=30
                self.ids[str('b'+str(x+1))].text=str('[b]Document '+str(x+1)+'          File = '+str(idss[temp[1]])+'          Score = '+str(temp[0])+'[/b]')
                self.ids[str('b'+str(x+1))].markup=True
                self.l.append(rev)
                
            if(no_results==0):
                self.ids['status'].text="[color=ff3333]NO RESULTS FOUND[/color]"
        t5=time.time()
        print(t5-t4)
       
        
        
        k_best.clear()
        val.clear()
    
runTouchApp(ScrollableLabel())