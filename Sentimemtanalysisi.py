# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 18:09:18 2024
https://github.com/bentrevett/pytorch-sentiment-analysis/blob/main/1%20-%20Neural%20Bag%20of%20Words.ipynb
@author: NITR_CS_PL4K
"""
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import nltk
#nltk.download('punkt')
import pandas as pd
import xml.etree.ElementTree as ET
import requests
import re
import collections
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm
#%%
# Fetch XML data from the provided link
response = requests.get("https://news.google.com/rss/search?q=green%20hydrogen&hl=en-IN&gl=IN&ceid=IN:en")

# Parse XML data
root = ET.fromstring(response.text)

# Initialize lists to store data
titles = []
links = []
param_links = []
descriptions = []
dates = []

# Iterate through each item in the XML
for item in root.findall('.//item'):
    # Get title
    title = item.find('title').text
    
    # Check if the title contains "green hydrogen" in any case
    if "green hydrogen" in title.lower():
        # Get link
        link = item.find('link').text
        
        # Get parameterized link
        param_link = item.find('guid').text
        
        # Get description
        description = item.find('description').text
        
        
        # Remove anchor tag content from description
        description = re.sub(r'<a.*?>', '', description)
        description = re.sub(r'</a*?>', '', description)
        description = re.sub(r'<font.*?>.*?</font>', '', description)  # Remove content within font tags
        description = re.sub(r'&amp;', '&', description)  # Replace &amp; with &
        description = re.sub(r'&\w+?;', '', description)  # Remove other HTML entities
        
        # Get date
        date = item.find('pubDate').text
        
        # Append data to lists
        titles.append(title)
        links.append(link)
        param_links.append(param_link)
        descriptions.append(description)
        dates.append(date)

# Create DataFrame
df = pd.DataFrame({
    'Title': titles,
    'Link': links,
    'Param_Link': param_links,
    'Description': descriptions,
    'Date': dates
})

# Save DataFrame as CSV
df.to_csv('output.csv', index=False)
# %%
# Display DataFrame
print("DataFrame saved as output.csv")
print(df)

df['Description_title'] = df['Title'] + ' - ' + df['Description']

# Print the updated DataFrame
print(df)
df['Description_title'][0]

# %%
df['Description_title_tokenized'] = df['Description_title'].apply(word_tokenize)

# Split the data into train and test sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Example output
print("Train DataFrame:")
print(train_data.head())
print("\nTest DataFrame:")
print(test_data.head())

#%%
min_freq = 5
special_tokens = ["<unk>", "<pad>"]

vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)


#%%
len(vocab)

#%%
vocab.get_itos()[:10]

#%%
def numericalize_example(example, vocab):
    ids = vocab.lookup_indices(example["tokens"])
    return {"ids": ids}

#%%
df = df.map(numericalize_example, fn_kwargs={"vocab": vocab})
train_data = train_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
test_data = test_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
#%%

#%%
train_data = train_data.with_format(type="torch", columns=["ids", "label"])
df = df.with_format(type="torch", columns=["ids", "label"])
test_data = test_data.with_format(type="torch", columns=["ids", "label"])

#%%


#%%


#%%


#%%




