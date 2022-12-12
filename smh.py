import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.title('IMDB Top 250 Movies')
st.header(
    'This dashboard will present the information about movies, which were made over a century period between 1921 and 2022')
st.markdown('''Let's display my dataset''')
st.code('''df = pd.read_csv('movies.csv)
df''')
df = pd.read_csv('movies.csv')
df
st.subheader('Features:')
st.markdown('''
rank - Movie Rank as per IMDB rating \n
id - Movie ID\n
name - Name of the Movie\n
year - Year of movie release\n
imdb_votes - Number of people who voted for the IMDB rating\n
imdb_rating - Rating of the Movie\n
certificate - Movie Certification\n
duration - Duration of the Movie\n
genre - Genre of the Movie\n
cast_id - ID of cast memeber who have worked on the movie''')
st.text("")
st.text("")

st.markdown("""---""")
st.subheader(
    '''Let's display some statistics, which includes mean, median, standard deviation and some other information about numerical fields''')
st.markdown('''There is some information about rank''')
st.metric(label='median', value=df['rank'].median())
st.dataframe(df['rank'].describe())
st.code('''print(df['rank'].describe())
print(df['rank'].median())''')

st.text("")
st.text("")
st.text("")
st.markdown('''Some information about year''')
st.metric(label='median', value=df['year'].median())
st.dataframe(df['year'].describe())
st.code('''print(df.year.describe())
print(df.year.median())''')

st.text("")
st.text("")
st.text("")
st.markdown('''Some information about imbd_votes''')
st.metric(label='median', value=df['imbd_votes'].median())
st.dataframe(df['imbd_votes'].describe())
st.code('''print(df.imbd_votes.describe())
print(df.imbd_votes.median())''')

st.text("")
st.text("")
st.text("")
st.markdown('''Some information about imdb_rating''')
st.metric(label='median', value=df['imdb_rating'].median())
st.dataframe(df['imdb_rating'].describe())
st.code('''print(df.imdb_rating.describe())
print(df.imdb_rating.median())''')

st.text("")
st.text("")
st.markdown("""---""")
st.subheader('In this step I check if my dataset has any NaN')
st.metric(label='NaN', value=df.isnull().sum().sum())
st.markdown('As we can see, my dataset has any NaN')
print(df.isnull().sum().sum())
st.code('df.isnull().sum().sum()')

st.markdown('And I wanna to delete these one')
st.code('df = df.dropna()')
df = df.dropna()
st.text("")
st.text("")
st.subheader('Now I check if my dataset has any NaN again')
st.metric(label='NaN', value=df.isnull().sum().sum())
print(df.isnull().sum().sum())
st.markdown('Now we have not any NaN in the dataset')

st.text("")
st.text("")
st.text("")
st.markdown("""---""")
st.header('Plots:')
st.subheader(
    'Now I make simple plots of different types for some numerical data. In order to introduce hypotheses, I want to look at a couple of dependencies.')
st.markdown(
    'First of all, consider the dependence of the rating of movie by year to understand the movies of which years people like the most')
st.metric(label='correlation', value=df['year'].corr(df['imdb_rating']))
print(df['year'].corr(df['imdb_rating']))
df['year'] = df['year'].astype('int')
fig, ax = plt.subplots()
sns.lineplot(
    x="year",
    y="imdb_rating",
    data=df,
    ax = ax,
)
st.pyplot(fig)
print(df['year'].corr(df['imdb_rating']))
st.code('''df['year'] = df['year'].astype('int')
ax = sns.lineplot(
    x="year",
    y="imdb_rating",
    data=df
)
print(df['year'].corr(df['imdb_rating']))''')
st.markdown('As we can see, the graph reaches its peak between 1960-1980. But there is no definite dependence; as evidence, next to the graph, its correlation is written, which is very small.')

st.markdown("""---""")
st.markdown('Now consider the dependence of the rating on the number of votes')
st.metric(label='correlation', value=df['imbd_votes'].corr(df['imdb_rating']))
print(df['imbd_votes'].corr(df['imdb_rating']))
fig, ax = plt.subplots()
sns.regplot(
    x="imbd_votes",
    y="imdb_rating",
    data=df,
    ax = ax,
)
st.pyplot(fig)
print(df['imbd_votes'].corr(df['imdb_rating']))
st.code('''ax = sns.lmplot(
    x="imbd_votes",
    y="imdb_rating",
    data=df
)
print(df['imbd_votes'].corr(df['imdb_rating']))''')
st.markdown('From the graph, you can see a positive dependence, that is, the higher number of votes strongly correlates with the high rating. This information can help in making hypotheses')

st.markdown("""---""")
st.markdown('And consider the dependence of the rating on the year of the duration of the movie to understand the average value')
df['duration'] = df['duration'].astype('int')
st.metric(label='correlation', value=df['duration'].corr(df['imdb_rating']))
print(df['duration'].corr(df['imdb_rating']))
fig, ax = plt.subplots()
sns.scatterplot(
    x="duration",
    y="imdb_rating",
    data=df,
    ax = ax,
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
st.pyplot(fig)
print(df['duration'].corr(df['imdb_rating']))

st.code('''df['duration'] = df['duration'].astype('int')
ax = sns.scatterplot(
    x="duration",
    y="imdb_rating",
    data=df
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
print(df['duration'].corr(df['imdb_rating']))''')
st.markdown('This graph also shows the dependence (correlation also confirms this for us) and we can see that movies that run no more than two hours are highly rated.')

st.markdown("""---""")
st.header('Hypothesis:')
st.subheader('The first hypothesis: in 2005-2010, drama was rated higher than action. And I want to check this hypothesis')
st.markdown('''Firstly, let's find the average rating of drama for the period that we need''')
drama_df = df[(df['genre'].str.contains('Drama')) & (df['year'] >= 2000)].groupby('year')
st.table(drama_df['imdb_rating'].mean())
st.code('''drama_df = df[(df['genre'].str.contains('Drama')) & (df['year'] >= 2000)].groupby('year')
drama_df['imdb_rating'].mean()''')
st.text("")
st.text("")
st.text("")
st.text("")
st.markdown('''Secondly, let's find the average rating of action for the period that we need too''')
action_df = df[(df['genre'].str.contains('Action')) & (df['year'] >= 2000)].groupby('year')
st.table(action_df['imdb_rating'].mean())
st.code('''action_df = df[(df['genre'].str.contains('Action')) & (df['year'] >= 2000)].groupby('year')
action_df['imdb_rating'].mean()''')
st.text("")
st.text("")
st.text("")
st.text("")
st.markdown('And now I combine the obtained values into one graph, which will show us the average rating of both genres for the period of time which I specified.')

fig, ax = plt.subplots()
ax.plot(drama_df['imdb_rating'].mean(), label="drama")
ax.plot(action_df['imdb_rating'].mean(), label="action")
ax.legend()
plt.show()
st.pyplot(fig)
st.code('''plt.plot(drama_df['imdb_rating'].mean(), label="drama")
plt.plot(action_df['imdb_rating'].mean(), label="action")
plt.legend()
plt.show()''')
st.markdown('As the graph shows, for the period 2005-2010 action movies were the most highly rated. This means that the hypothesis was invalid, and we proved it!')

st.markdown("""---""")
st.subheader('The second hypothesis: since 2000, crime films have received more votes than mystery and fantasy, that is, they are in greater demand among the rest. To prove this, I chose three years, when movies of all three genres were released (2003, 2006 and 2021), for which I will compare their quantity of votes.')
st.markdown('''Firstly, let's find the average number of votes among crime films for the given three years''')
crime_df = df[(df['genre'].str.contains('Crime')) & (df['year'].isin([2003, 2006, 2021]))].groupby('year')
st.dataframe(crime_df['imbd_votes'].mean())
st.code('''crime_df = df[(df['genre'].str.contains('Crime')) & (df['year'].isin([2003, 2006, 2021]))].groupby('year')
crime_df['imbd_votes'].mean()''')
st.text("")
st.text("")
st.text("")
st.text("")
st.markdown('Next, we find the average number of votes among mystery films for the given three years')
mystery_df = df[(df['genre'].str.contains('Mystery')) & (df['year'].isin([2003, 2006, 2021]))].groupby('year')
st.dataframe(mystery_df['imbd_votes'].mean())
st.code('''mystery_df = df[(df['genre'].str.contains('Mystery')) & (df['year'].isin([2003, 2006, 2021]))].groupby('year')
mystery_df['imbd_votes'].mean()''')
st.text("")
st.text("")
st.text("")
st.text("")
st.markdown('Then we find the average number of votes among fantasy films for the given three years')
fantasy_df = df[(df['genre'].str.contains('Fantasy')) & (df['year'].isin([2003, 2006, 2021]))].groupby('year')
st.dataframe(fantasy_df['imbd_votes'].mean())
st.code('''fantasy_df = df[(df['genre'].str.contains('Fantasy')) & (df['year'].isin([2003, 2006, 2021]))].groupby('year')
fantasy_df['imbd_votes'].mean()''')

st.markdown('''And now, having all the information we need, let's build a graph that will show the average number of votes for all three genres of movies for the period I specified''')
fig, ax = plt.subplots()
N = 3
ind = np.arange(N)
width = 0.25
xvals = crime_df['imbd_votes'].mean()
bar1 = plt.bar(ind, xvals, width, color='c')
yvals = mystery_df['imbd_votes'].mean()
bar2 = plt.bar(ind + width, yvals, width, color='cyan')
zvals = fantasy_df['imbd_votes'].mean()
bar3 = plt.bar(ind + width * 2, zvals, width, color='black')
plt.xlabel("Years")
plt.ylabel('Votes')
plt.title("Votes over 3 years")
plt.xticks(ind + width, ['2003', '2006', '2021'])
plt.legend((bar1, bar2, bar3), ('Crime', 'Mystery', 'Fantasy'))
plt.show()
st.pyplot(fig)
st.code('''N = 3
ind = np.arange(N)
width = 0.25

xvals = crime_df['imbd_votes'].mean()
bar1 = plt.bar(ind, xvals, width, color='c')

yvals = mystery_df['imbd_votes'].mean()
bar2 = plt.bar(ind + width, yvals, width, color='cyan')

zvals = fantasy_df['imbd_votes'].mean()
bar3 = plt.bar(ind + width * 2, zvals, width, color='black')

plt.xlabel("Years")
plt.ylabel('Votes')
plt.title("Votes over 3 years")

plt.xticks(ind + width, ['2003', '2006', '2021'])
plt.legend((bar1, bar2, bar3), ('Crime', 'Mystery', 'Fantasy'))
plt.show()''')

st.markdown('Looking at the graph, we can see that the hypothesis is partially true, that is, movies of crime were the most popular in 2006, but in 2003 and 2021 fantasy became the most relevant genre.')
st.markdown("""---""")
st.header('Data transformation:')
st.subheader('Because of I had to do data cleanup, I will make one new column by modifying data from other columns. Namely I want to convert text data with information about certification to numerical one, starting with movies for all ages to 18+')
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
st.code('''def certificate_to_number(row):
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
df''')