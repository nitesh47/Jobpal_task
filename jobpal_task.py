#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 17:50:45 2019

@author: nitesh
"""


'''Importing the libraries '''

import json
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sn
import numpy as np
from nltk.corpus import stopwords
from scipy.sparse import hstack
from bs4 import BeautifulSoup
from sklearn.linear_model import SGDClassifier

class task(object):
    
    def __init__(self,data):
        self.data = data
        self.X_train = []
        self.pred = []
        self.space_removal = re.compile('[/(){}\[\]\|@,;]')
        self.symbol_removal = re.compile('[^0-9a-z #+_]')
        self.stopwords = set(stopwords.words('english'))
        self.lb_encode = LabelEncoder()
        self.count_vec = CountVectorizer()
        self.tfidf = TfidfTransformer()
     
    ''' Splitting the data into training set'''
    
    def train_data(self, data):
        for i in range(141):
            dict_io = {}
            dict_io['description'] = data[i]['description']
            dict_io['title'] = data[i]['title']
            dict_io['level'] = data[i]['level']
            self.X_train.append(dict_io)
        return self.X_train
    
    ''' Splitting the data into prediction set''' 
    
    def pred_data(self, data):
        for i in range(141,216):
            dict_io = {}
            dict_io['description'] = data[i]['description']
            dict_io['title'] = data[i]['title']
            self.pred.append(dict_io)
        return self.pred
    
    ''' Text preprocessing'''
    
    def clean_text(self,text):
        text = BeautifulSoup(text, "lxml").text
        text = text.lower()
        text = self.space_removal.sub(' ', text) 
        text = self.symbol_removal.sub('', text)
        text = re.sub("\d+", " ", text)
        text = re.sub("outfitteryis", "outfittery is ", text)
        text = re.sub("missionwe", " mission we ", text)
        text = re.sub("successyour", " success your ", text)
        text = re.sub("tasksour", " tasks our ", text)
        text = text.split()
        text = [w for w in text if w not in self.stopwords]
        text = [w for w in text if len(w)>1]
        text = ' '.join(text)
        return text
    
    '''Encoding categorical data'''
    
    def label_encoder(self, labels):
        
        y = self.lb_encode.fit_transform(labels)
        print(self.lb_encode.classes_)
        return y
    
    '''Inverse transforming categorical data'''
    
    def inverse_transform_label(self,y_pred):
        y_pred = self.lb_encode.inverse_transform(y_pred)
        return y_pred
    
    '''
        Fit Countervector and Tfidf on the training dataset. '''
        
    def ConterVec_fit_tfidf(self, data):
        vect_data = self.count_vec.fit_transform(data)
        fit_tfidf = self.tfidf.fit_transform(vect_data)
        return fit_tfidf
    
    '''  Using the transform method of the same object on 
    testing data to create feature representation of test data.  '''
    
    def ConterVec_tfidf_transform(self, data):
        vect_data = self.count_vec.transform(data)
        transform_tfidf = self.tfidf.transform(vect_data)
        return transform_tfidf
    
    ''' Stacking  two sparse matrix '''
    def stack_train_features(self, X, X1):
        stacked_X = hstack([X, X1])
        return stacked_X
    
        
    ''' Model initialization '''    
    def model_config(self):
        sgd =SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)
        return sgd
        
    
    
if __name__ == "__main__":
    
    ''' Importing the dataset ''' 
    
    with open('data.json') as f:
        data = json.load(f)
        
    A = task(data)
    
    ''' data divided into a training set and prediction set'''
    train = A.train_data(data)
    pred = A.pred_data(data)
    
    ''' Training and Prediction data set converted into DataFrame '''
    train_df = pd.DataFrame(train)
    pred_df = pd.DataFrame(pred)
    
    ''' plot classes in training set'''
    plt.figure(3,figsize =(40,40))
    fig = plt.figure(figsize=(8,6))
    plt.ylabel('Counts', fontsize=13)
    train_df.groupby('level').description.count().plot.bar(ylim=0)
    plt.show()
    
    ''' Cleaned Training and Prediction Data set'''
    train_df['description'] = train_df['description'].apply(A.clean_text)
    train_df['title'] = train_df['title'].apply(A.clean_text)
    pred_df['preprocessed_description'] = pred_df['description'].apply(A.clean_text)
    pred_df['preprocessed_title'] = pred_df['title'].apply(A.clean_text)
    
    
    ''' Training and prediction feature sets '''
    X_description = A.ConterVec_fit_tfidf(train_df.description)
    pred_discription = A.ConterVec_tfidf_transform(pred_df.preprocessed_description)
    
    X_title = A.ConterVec_fit_tfidf(train_df.title)
    pred_title = A.ConterVec_tfidf_transform(pred_df.preprocessed_title)
    
    ''' Stacking description and title features into a single matrix '''
    X_stacked_features = A.stack_train_features(X_description, X_title)
    pred_stacked_features = A.stack_train_features(pred_discription, pred_title)
    
    print(X_stacked_features.shape,pred_stacked_features.shape)
#    
    
    ''' Labels Encoding '''
    labels = list(train_df['level'])
    y = A.label_encoder(labels)
    
    ''' Split train and test data in the ratio of 90:10 '''
    X_train, X_test, y_train, y_test = train_test_split(X_stacked_features, y, 
                                                    test_size = 0.1, 
                                                    random_state = 10,
                                                    stratify=y)
    
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    ''' Model building initialization'''
    sgd = A.model_config()
    
    sgd.fit(X_train, y_train)
    
    '''Prediction on test dataset'''
    y_pred = sgd.predict(X_test)
    #print(A.inverse_transform_label(y_pred))
    
    print(classification_report(y_test, y_pred))
    #print(confusion_matrix(y_test, y_pred))
    
    ''' Prediction on missing data '''
    pred_predict = sgd.predict(pred_stacked_features)
    pred_predict = A.inverse_transform_label(pred_predict)
    #print(pred_predict)
    
    array = np.array(confusion_matrix(y_test, y_pred))
    df_cm = pd.DataFrame(array, index = [i for i in ['Entry Level', 'Internship', 'Mid Level',
                             'Senior Level']],
                  columns = ['Entry Level', 'Internship', 'Mid Level',
                             'Senior Level'])
    plt.figure(figsize = (8,8))
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})
    
    
    pred_df['level'] = pred_predict
    
    ''' Prediction Result '''
    pred_df = pred_df[['level','description','title']]
    with open('Predicted_result1.json', 'w') as f:
        f.write(pred_df.to_json(orient='records', lines=True))
