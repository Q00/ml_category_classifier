import csv
import nltk
import json
import numpy as np
import gensim
import tensorflow as tf 
from gensim.models import Word2Vec
import pandas as pd


train_filepath = "train.csv"
train_df = pd.read_csv(train_filepath,header=None, names=['category','content'])
category_list = {'tech' : 0, 'business' : 1, 'sport' : 2, 'entertainment' : 3, 'politics' : 4}
token =[]
embeddingmodel = []
data_df = train_df
parsed_data_df=pd.DataFrame(columns=['id', 'category', 'term', 'type'])

for i in range(len(data_df.index)):
    part_data_df =data_df.iloc[i]
    category=part_data_df['category']
    content=part_data_df['content']
    part_parsed_data_df=pd.DataFrame(nltk.pos_tag(nltk.word_tokenize(content)), columns=['term','type'])
    part_parsed_data_df['category']=category
    part_parsed_data_df['id']=i
    part_parsed_data_df=part_parsed_data_df[['id','category','term','type']]
    parsed_data_df=pd.concat([parsed_data_df, part_parsed_data_df])


def plot_wordcloud(term_vec):
    
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud, STOPWORDS 
    wordcloud = WordCloud(max_font_size=200,
                          stopwords=STOPWORDS,background_color='#FFFFFF',width=1200,height=800)
    res_wordcloud=wordcloud.generate(' '.join(term_vec))
    plt.figure(figsize=(10,8))
    plt.imshow(res_wordcloud)
    plt.tight_layout(pad=0)
    plt.axis('off')


target_category=['politics']
target_type=['NN','NNP']
target_parsed_data_df=parsed_data_df.loc[parsed_data_df['category'].isin(target_category)&
                                        parsed_data_df['type'].isin(target_type)]
term_vec=target_parsed_data_df['term']

plot_wordcloud(term_vec)

