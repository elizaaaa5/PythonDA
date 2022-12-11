#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


# Let's display my dataset
st.title('IMDB Top 250 Movies')
st.header('IMDB Top 250 Movies')
st.markdown('IMDB Top 250 Movies')
st.subheader('IMDB Top 250 Movies')
st.code('''df = pd.read_csv('movies.csv)
df''')

# In[2]:


df = pd.read_csv('movies.csv')
#df


# Let's display some statistics, which include  mean, median, standard deviation and some other information of numerical fields

# There are some information about rank

# In[3]:


print(df['rank'].describe())
print(df['rank'].median())


# Some information about year

# In[4]:


print(df.year.describe())
print(df.year.median())


# Some information about imbd_votes

# In[5]:


print(df.imbd_votes.describe())
print(df.imbd_votes.median())


# Some information about imdb_rating

# In[6]:


print(df.imdb_rating.describe())
print(df.imdb_rating.median())


# I decided to display information about data types in my dataset

# In[7]:


df.info()


# In this step I check if my dataset has some NaN

# In[8]:


df.isnull().sum().sum()


# As we can see, my dataset has some NaN

# And I wanna to delete these one

# In[9]:


df = df.dropna()


# Now I check if my dataset has some NaN again

# In[10]:


df.isnull().sum().sum()


# Now we have not any NaN in the dataset

# Plots:
# Now I make simple plots with different types for some numerical data

# In order to put any hypotheses, I want to look at a couple of dependencies.
# First of all, consider the dependence of the rating of movie by year to understand the movies of which years people like
# the most

# In[11]:


df['year'] = df['year'].astype('int')
ax = sns.lineplot(
    x="year",
    y="imdb_rating",
    data=df
)
print(df['year'].corr(df['imdb_rating']))


# As we can see, the graph reaches its peak between 1960-1980.
# But there is no definite dependence; as evidence, next to the graph, its correlation is written, which is very small.

# 
# Now consider the dependence of the rating on the number of votes

# In[12]:


ax = sns.lmplot(
    x="imbd_votes",
    y="imdb_rating",
    data=df
)
print(df['imbd_votes'].corr(df['imdb_rating']))


# On the graph, you can see a positive dependence, that is, the number of votes strongly affects the high rating.
# This information can help in making hypotheses.

# And consider the dependence of the rating on the year of the duration of the movie to understand the average value

# In[13]:


df['duration'] = df['duration'].astype('int')
ax = sns.scatterplot(
    x="duration",
    y="imdb_rating",
    data=df
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
print(df['duration'].corr(df['imdb_rating']))


# This graph also shows the dependence (correlation also confirms this for us) and we can see that movies
# that run no more than two hours are highly rated.

# Hypothesis:

# The first hypothesis: it is believed that in 2005-2010, drama were rated higher than action. And I want to check this hypothesis

# Firstly, let's find the average rating of drama for the period that we need

# 

# In[14]:


drama_df = df[(df['genre'].str.contains('Drama')) & (df['year'] >= 2000)].groupby('year')
drama_df['imdb_rating'].mean()


# Secondly, let's find the average rating of action for the period that we need too

# In[15]:


action_df = df[(df['genre'].str.contains('Action')) & (df['year'] >= 2000)].groupby('year')
action_df['imdb_rating'].mean()


# And now I combine the obtained values into one graph, which will show us the average rating of both genres for the period of time
# which I specified.

# In[16]:


plt.plot(drama_df['imdb_rating'].mean(), label="drama")
plt.plot(action_df['imdb_rating'].mean(), label="action")
plt.legend()
plt.show()


# As the graph shows, for the period 2005-2010 action  were the most highly rated. This means that the hypothesis was invalid,
# and we proved it!

# The second hypothesis: it is believed that since 2000, crime films have received more votes than mystery and fantasy,
# that is, they are in greater demand among the rest. For proof this, I chose three years, when there were coincidences in releases
# (2003, 2006 and 2021), for which I will compare.

# Firstly, let's find the average number of votes among crime for the given three years.

# In[17]:


crime_df = df[(df['genre'].str.contains('Crime')) & (df['year'].isin([2003, 2006, 2021]))].groupby('year')
crime_df['imbd_votes'].mean()


# Next, we find the average number of votes among mystery for the given three years

# In[18]:


mystery_df = df[(df['genre'].str.contains('Mystery')) & (df['year'].isin([2003, 2006, 2021]))].groupby('year')
mystery_df['imbd_votes'].mean()


# Then we find the average number of votes among fantasy for the given three years

# In[19]:


fantasy_df = df[(df['genre'].str.contains('Fantasy')) & (df['year'].isin([2003, 2006, 2021]))].groupby('year')
fantasy_df['imbd_votes'].mean()


# And now, having all the information we need, let's build a graph that will show the average number of votes for all
# three genres of movies for the period I specified

# In[20]:


N = 3
ind = np.arange(N)
width = 0.25

xvals = crime_df['imbd_votes'].mean()
bar1 = plt.bar(ind, xvals, width, color='r')

yvals = mystery_df['imbd_votes'].mean()
bar2 = plt.bar(ind + width, yvals, width, color='c')

zvals = fantasy_df['imbd_votes'].mean()
bar3 = plt.bar(ind + width * 2, zvals, width, color='m')

plt.xlabel("Years")
plt.ylabel('Votes')
plt.title("Votes over 3 years")

plt.xticks(ind + width, ['2003', '2006', '2021'])
plt.legend((bar1, bar2, bar3), ('Crime', 'Mystery', 'Fantasy'))
plt.show()


# Looking at the graph, we can see that the hypothesis is partially true, that is, movies of crime were the most popular
# in 2006, but in 2003 and 2021 fantasy became the most relevant genre.

# Data transformation: Because of I had to do data cleanup, I will make one new column by modifying data from other columns.
# Namely I want to convert text data with information about certification to numerical one, starting with movies for all ages to 18+

# In[21]:


def certificate_to_number(row):
    letter = row['certificate']
    if letter == 'G':
        return 1
    if letter == 'U' or letter == '7' or letter == 'Approved' or letter == 'PG':
        return 2
    if letter == 'Passed' or letter == '12+' or letter == '13':
        return 3
    if letter == '15+':
        return 4
    if letter == 'UA' or letter == '16':
        return 5
    if letter == 'R':
        return 6
    if letter == 'A' or  letter == '18':
        return 7
    if letter == 'Not Rated':
        return 0


df['certificate_numeric'] = df.apply(certificate_to_number, axis=1)
# df['certificate_numeric'] = df['certificate_numeric'].astype('int')
df

