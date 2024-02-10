# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import xml.etree.ElementTree as ET
import requests

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
df.to_csv(r'F:/output.csv', index=False)

# Display DataFrame
print("DataFrame saved as output.csv")
print(df)
