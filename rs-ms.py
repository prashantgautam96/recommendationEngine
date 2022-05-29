import string
from cv2 import split
from matplotlib.pyplot import title
import numpy as np
import pandas as pd
# ____________________________
import pandas as pd
import numpy as np
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







# ___Extracting Data________________________________________________________________

# creating the dataframe with the name movies and credits
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# merging two df so that we can access the the property which is required to make model
movies= movies.merge(credits,on='title')

# doing operations on the year column of the datframe, to extract the required data format
movies['year'] = pd.to_datetime(movies['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

# creating the dataframe with the name c_ratings for a diffrent dataset
c_ratings = pd.read_csv('Collaborative Filtering Dataset/dataset/ratings.csv')
c_movies = pd.read_csv('Collaborative Filtering Dataset/dataset/movies.csv')

# merging the c_ratings ,c_movies dataframe to use properties of both together, 
# and dropping the genres and timestamp colums as we don't need this data as per requirement of the model
c_ratings = pd.merge(c_movies,c_ratings).drop(['genres','timestamp'],axis=1)
# print(ratings.shape)
# ratings.head()
# _________________________________________________________________________________




# _________________________________TAGLINE_________________________________________________
# print(movies['tagline'])
# ___________________________________________________________________________________________






# ____________________________________________________________________________weighted_average_technique___________________________________________________________________


# In this section i have written code for the weightage average technique in which we take the movie dataset in which we are 
# calculating the weighatge average for each movie's average rating
# ____________________using formula
# W=((R*V)+(C*m))/(V+m)
# where:
# W = Weighted Rating
# R = Average for the movie as a number from 0 to 10(mean)=(Rating)
# V= Number of votes for the movie = (votes)
# m = minimum votes required to be part in the recommendation as we can't recommend a movie to the user which have a very good voting average but it lacks in the number
# of people who have rated it as there there is possibilities that very few people liked and given good rating which comes out to a high average rating but most of the people don't like
# that movie
# C = mean vote across the whole report


# storing the movies dataframe in rmovies so that we can use it without affecting the original dataframe as we have to use movies datframe for other model also
rmovies = movies


V= rmovies['vote_count']  
R= rmovies['vote_average']
C= rmovies['vote_average'].mean()
# i will only  consider a movie for recommendation which have more than 70 percentile vote for that i have used quantile to take limit to be listed in the recommendations
m= rmovies['vote_count'].quantile(0.70)


# applying the above logic for calculating the weighted average as creating a new column ['weighted_average'] in dataframe to store values
rmovies['weighted_average']=((R*V)+(C*m))/(V+m)

# Sorting the whole dataframe by using the values of ['weighted_average'] in the descending order so that we can 
# get the best recommendations at top
movies_sorted_ranking= rmovies.sort_values('weighted_average',ascending=False)
msr=movies_sorted_ranking[['movie_id','title','overview','genres','keywords','year','cast','crew','vote_count','vote_average','weighted_average','popularity']].head(20)


# removing the unneccsary colums__, and using the below only for the dataframe
# ----------------------------
# genres
# id
# keywords
# title
# overview
# cast n crew
# ----------------------------


# Till now we were using only the vote count and the weight average for recommending but as in our dataframe there is one more column ,
# "popularity", we will be going to use that as there may be some movies which users may not be seen or visited, so we shoul not skipp those populer movies
# so here i will give the 50 percent importance to weighted average and 50 percent to the popularity for recommendation purpose
# so before proceeding with the diving all the weightage for model i will use MinMaxScaler to scale-down the values of  ['weighted_average','popularity']  
# in the range from 0 to 1 as in the data frame weighted_average and popularity have diffrent range of magnitudes 
from sklearn.preprocessing import MinMaxScaler

scaling=MinMaxScaler()
movies_scaled_df = scaling.fit_transform(rmovies[['weighted_average','popularity']])

# creating the movies_normalised_df by combining the scaled datframe with 'weighted_average','popularity'
movies_normalised_df= pd.DataFrame(movies_scaled_df,columns=['weighted_average','popularity'])

# adding the two columns ['normalised_weight_average','normalised_popularity'] in our parent datframe rmovies and storing the values of movies_normalised_df
rmovies[['normalised_weight_average','normalised_popularity']]=movies_normalised_df

# creating a score column and  storing the values  using 50 percentage weightage to each 'normalised_weight_average','normalised_popularity'
rmovies['score']= rmovies['normalised_weight_average']*0.5 + rmovies['normalised_popularity']*0.5

# sorting the values of the dataframe according to the score in descending order
rmovies= rmovies.sort_values(['score'],ascending=False)
rmovies=rmovies[['title','movie_id','overview','genres','vote_count','vote_average','keywords','year','cast','crew','weighted_average','popularity','normalised_weight_average','normalised_popularity','score']].head(20)
# __________________________________________________________________________________________________________________________________________________________________________________________________________________________________________






# ____________________________________________________________________COSINE SIMILARITY______________________________________________________________________________________________________________________________________________________________________


# ----missing data
# checking the null data
# print(movies.isnull().sum())
# movies.dropna(inplace=True)
# print(movies.shape,"movie1")
# ---check duplicaye data
# print(movies.duplicated().sum())
movies.iloc[0].genres
# [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]
# ['Action','Adventure','Fantasy','SciFi']


# converting the string of list into list using ast
import ast

# a function to convert the data into list
def convert(obj):
    L= []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# applying the function convert
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)



# function to extract casts from the dataframe
def convert3(obj):
    L= []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter!= 3:
            L.append(i['name'])
            counter+=1
        else:
            break         
    return L

# applying the function
movies['cast'] = movies['cast'].apply(convert3)




# function to fetch the directer name from the datframe
def fetch_director(obj):
    L= []

    for i in ast.literal_eval(obj):
        if i['job']== 'Director':
                L.append(i['name'])
                break
             
    return L




movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x:str(x).split())
# removing all the spaces in a string
movies['genres']= movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']= movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']= movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']= movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
# print(movies.shape,"movie")

# making a new colums to contgnet all 4 properties
movies['tags'] =  movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# making the new dataframe  as new_df and storing all three columns movie_id, title and tags of movies
new_df = movies[['movie_id','title','tags']]

# in the tags if in any string there is space then we will remove that space and join that so that they don't come out as two diffrent movies
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))

# after that converting all the string into thr lower characters
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# Here we are going to do the filetring using the tags as how much similarity is between two tags will be the deciding factor 
# that how much similarity is between the both movie as tags have all the properties about a particuler movie associated with it

import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags']=new_df['tags'].apply(stem)


# VECTORISATION____________________________
# we have to find similarity using tags
# text_vectorisation_____using Bag of words
# so firstly we will convert the tags of each movie to vector
# so if we want to recommend the movie which have vectors closest to the given words
# ..bag of words
# we combine all the tags->most used words

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()
# print(vectors)


# we will calculate the cosine distance between each other vector
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)

similarity


# __________________________________________________________________________________________________________________________________________________________
# creating a function so that we can recommend the movie

def recommend(movie):
    # taking the index of the movie from the given input
    movie_index = new_df[new_df['title'] == movie].index[0]
    # calculting distances between the vectors
    distances = similarity[movie_index]

    # sorting and storing the values of similer movies in movieslist
    movies_list = sorted(list(enumerate(distances)),reverse=True,key = lambda x: x[1])[1:6]
    

    recommended_movie = []
    recommended_movie_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        
        recommended_movie.append(movies.iloc[i[0]].title)
        # fetch poster from api= d3692ce4fecfce91e8bcc1676841d69d
        # recommended_movie_posters.append(fetch_poster(movie_id))
    return recommended_movie

print(recommend("Avatar"))




# _________________________________________________________________________GENRES BASED_____________________________________________________________________

# through this section we are recommending the movie based on the genres the user want to see


s = movies.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_movies = movies.drop('genres', axis=1).join(s)

def build_chart(genre, percentile=0.85):
    df = gen_movies[gen_movies['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    # creating the new datframe qualified to seprate the movies which have vote count more then or equal to our limit and 
    # and take those which have certain votes and vote average, as this will not be good 
    # that we are recommending a movie on the top on which any user have not rated or voted
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    # now doing the following calculations to find ranking 
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)

    # sorting all the values with respect to wr
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified


# _______________________________________________________________________________________________________________________________________________________________________




# _____________________________________________________________________________COLLABARATIVE FILTERING_________________________________________________________________________________________

userRatings = c_ratings.pivot_table(index=['userId'],columns=['title'],values='rating')
userRatings.head()
# print("Before: ",userRatings.shape)
userRatings = userRatings.dropna(thresh=10, axis=1).fillna(0,axis=1)
#userRatings.fillna(0, inplace=True)
# print("After: ",userRatings.shape)

corrMatrix = userRatings.corr(method='pearson')
corrMatrix.head(100)


def get_similar(movie_name,rating):
    similar_ratings = corrMatrix[movie_name]*(rating-2.5)
    similar_ratings = similar_ratings.sort_values(ascending=False)
    #print(type(similar_ratings))
    return similar_ratings


# romantic_lover = [("(500) Days of Summer (2009)",5),("Alice in Wonderland (2010)",3),("Aliens (1986)",1),("2001: A Space Odyssey (1968)",2)]



def recommendr(fMovie,fRating):
    action_lover = [(fMovie,fRating)]
    similar_movies = pd.DataFrame()
    for movie,rating in action_lover:
        similar_movies = similar_movies.append(get_similar(movie,rating),ignore_index = True)

    print(similar_movies.sum().sort_values(ascending=False).head(10))


recommendr("(500) Days of Summer (2009)",5)
# ________________________________________________________________________________________________________________________________________________________________________________________________





# _________________________________________________________________SENDING DATA USING PKL__________________________________________________________________________________


import pickle
pickle.dump(new_df,open('movies.pkl','wb'))

pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


pickle.dump(rmovies,open('emovies.pkl','wb'))
pickle.dump(gen_movies,open('gen_movies.pkl','wb'))
pickle.dump(c_ratings,open('c_ratings.pkl','wb'))

# _____________________________________________________________________________________________________________________________________________________________________________