# imports

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
import warnings; warnings.simplefilter('ignore')

# loading movie_metadata file and categorizing by genre
movie_data = pd.read_csv('sample_data/movies_metadata.csv')
movie_data.head()

movie_data['genres'] = movie_data['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

'''
top movies chart by calculating the weighted rating
v is the number of votes for the movie
min_votes is the minimum votes required to be listed in the chart
avg_rating is the average rating of the movie
c is the mean vote across the whole report
'''

vote_counts = movie_data[movie_data['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = movie_data[movie_data['vote_average'].notnull()]['vote_average'].astype('int')
c = vote_averages.mean()
c
min_votes = vote_counts.quantile(0.95)
min_votes

# categorizing based on release date
movie_data['year'] = pd.to_datetime(movie_data['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

# for a movie to feature in the charts, it must have more votes than at least 95% of the movies in the list.
qualified = movie_data[(movie_data['vote_count'] >= min_votes) & (movie_data['vote_count'].notnull()) & (movie_data['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
qualified.shape

def weighted_rating(x):
    v = x['vote_count']
    avg_rating = x['vote_average']
    return (v/(v+min_votes) * avg_rating) + (min_votes/(min_votes+v) * c)
    
# top rated movies    
qualified['wr'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('wr', ascending=False).head(250)
qualified.head(15)

# function to build charts for particular genres
s = movie_data.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_movie_data = movie_data.drop('genres', axis=1).join(s)

def build_chart(genre, percentile=0.85):
    df = gen_movie_data[gen_movie_data['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    c = vote_averages.mean()
    min_votes = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= min_votes) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+movie) * x['vote_average']) + (movie/(movie+x['vote_count']) * c), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified
    
 # list movies from genre 'romance'
 build_chart('Romance').head(15)
