#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json
import os
import numpy as np


notebook_path = os.path.abspath("extract_selected.ipynb")
print("done")


# In[6]:


train_json = os.path.join(os.path.dirname(notebook_path), "train_v2.1.json")


df = pd.read_json(train_json)


#df['wellFormedAnswers'][50:100]
#df.iloc[0, 1]
#df.iloc[0,2]


# In[2]:


import re
import csv
qwords = ['who', 'what', 'when', 'where', 'why', 'how', 'does', 'can', 'is', 'did', 'was', 'are']

def clean_lowercase(text):
    return str(text).lower()

def clean_non_alpha_num(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def filter_by_question_word(text):
    text_split = text.split()
    if len(text_split) == 0 or text_split[0] not in qwords:
        #print("NO", text)
        return("")
    else:
        #print("YES", text)
        return text
    
def return_selected_passage(passages):
    #for passage in passages:
      #  print(passage)
    print(passages) 
    print("-----------")

train = df[0:200000].copy()
train = train.applymap(lambda s:s.lower() if type(s) == str else s)
train = train.applymap(lambda s: re.sub('[^a-zA-Z\D]', '', s) if type(s) == str else s)
#short['query'] = short['query'].apply(lambda s: s if s[0] in qwords else "")
#print(short['passages'].iloc[0])
train['query'] = train['query'].apply(clean_non_alpha_num)
train['query'] = train['query'].apply(filter_by_question_word)

test = df[200000:400000].copy()
test = test.applymap(lambda s:s.lower() if type(s) == str else s)
test = test.applymap(lambda s: re.sub('[^a-zA-Z\D]', '', s) if type(s) == str else s)
#short['query'] = short['query'].apply(lambda s: s if s[0] in qwords else "")
#print(short['passages'].iloc[0])
test['query'] = test['query'].apply(clean_non_alpha_num)
test['query'] = test['query'].apply(filter_by_question_word)


#short['passages'] = short['passages'].apply(return_selected_passage)

#print(short['query'])
#print(short['passages'][0:10])
# first = short['query'][0:100]
# print(first)

#for val in first:
 #   print(val)


# In[11]:


train.to_csv('MARCO_train_data.csv')
test.to_csv('MARCO_test_data.csv')


# In[7]:


data = pd.read_csv('./short_data.csv')


# In[40]:


test_string = data['passages'][0][1:-1]
print(test_string)
print()
test_string = test_string.replace("\"", "\'")
test_string = test_string.replace("\'is_selected\'", "\"is_selected\"")
test_string = test_string.replace("\'passage_text\'", "\"passage_text\"")
test_string = test_string.replace("\'url\'", "\"url\"")
test_string = test_string.replace("\"passage_text\": \'", "\"passage_text\": \"")
test_string = test_string.replace("\'}", "\"}")
test_string = test_string.replace("\', \"url\"", "\", \"url\"")
test_string = test_string.replace("\"url\": \'", "\"url\": \"")
print()
print(test_string)
test_ra = test_string.split(", {")
for i in range(len(test_ra)-1):
    test_ra[i+1] = "{" + test_ra[i+1]
print()
print(test_ra)


# In[3]:


def extract_selected_helper(passage):
    passage = passage[1:-1]
    passage = passage.replace("\"", "\'")
    passage = passage.replace("\'is_selected\'", "\"is_selected\"")
    passage = passage.replace("\'passage_text\'", "\"passage_text\"")
    passage = passage.replace("\'url\'", "\"url\"")
    passage = passage.replace("\"passage_text\": \'", "\"passage_text\": \"")
    passage = passage.replace("\'}", "\"}")
    passage = passage.replace("\', \"url\"", "\", \"url\"")
    passage = passage.replace("\"url\": \'", "\"url\": \"")

    passage_ra = passage.split(", {")
    for i in range(len(passage_ra)-1):
        passage_ra[i+1] = "{" + passage_ra[i+1]
    
    for i in range(len(passage_ra)):
        try:
            passage_dict = json.loads(passage_ra[i])
        except:
            print(passage_ra[i])
            continue
        if passage_dict['is_selected'] == 1:
            return passage_dict['passage_text']
   
    return '' #return empty string if no passage is 'selected'


# In[4]:


def extract_selected(csv_file):
    data = pd.read_csv(csv_file)
    for index, row in data.iterrows():
        data.at[index,'passages'] = extract_selected_helper(row['passages'])
    data.to_csv('extracted_' + csv_file)


# In[18]:


extract_selected('MARCO_train_data.csv')
extract_selected('MARCO_test_data.csv')


# In[6]:


import json

# Function to convert a CSV to JSON
# Takes the file paths as arguments
def make_json(csvFilePath, jsonFilePath):
     

    data = pd.read_csv(csvFilePath)
    data = data.rename(columns={"passages": "context", "query": "question"})
    data = data.astype({"context": str, "question": str, "answers": str})
    data["context_length"] = data["context"].str.split().str.len()
    data["answers_length"] = data["answers"].str.split().str.len()
    data["question_length"] = data["question"].str.split().str.len()
    data = data[data.context_length < 512]
    data = data[data.answers_length < 512]
    data = data[data.question_length < 512]
    data = data[data.answers != '[\'No Answer Present.\']']
    data = data[data.question != 'nan']
    # CHANGE THIS DEPENDING ON DESIRED SIZE OF TRAIN DATASET
    data = data[0:10000] 
    data['answers'] = data['answers'].str[2:-2]
    
    del data['Unnamed: 0.1']
    del data['query_id']
    del data['query_type']
    del data['wellFormedAnswers']
    del data['answers_length']
    del data['context_length']
    del data['question_length']
    del data['Unnamed: 0']

    data["answer_start"] = 0 # not used by BERT-SQG
    
    num_samples = len(data.index)
    max_query_length = data.question.map(lambda x: len(x.split())).max()
    max_answer_length = data.answers.map(lambda x: len(x.split())).max()
    max_context_length = data.context.map(lambda x: len(x.split())).max()

    print("num_samples: ", num_samples)
    print("max_seq_length: ", max(max_query_length, max_answer_length, max_context_length))
    print("max_answer_length: ", max_answer_length)
    print("max_query_length: ", max_query_length)
    
    result = data.to_json(orient="records")
    
    with open(jsonFilePath, 'w') as outfile:
            parsed = json.loads(result)
            json.dump(parsed, outfile)
    


# In[7]:


# Driver Code
 
# Decide the two file paths according to your
# computer system
csvFilePath = r'extracted_MARCO_train_data.csv'
jsonFilePath = r'MARCO_train_10K.json'
 
# Call the make_json function
make_json(csvFilePath, jsonFilePath)

# COPY AND MODIFY THIS CELL AS NEEDED TO CREATE TEST SPLITS OR MORE TRAINING DATA

