from audioop import reverse
from turtle import right, speed, width
from click import option
from matplotlib.pyplot import title
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from streamlit_option_menu import option_menu



# __________________________________________________________________NEW DF_COSINE_SIMLARITY_______________________________________________________________________________________________


# function to fetch the poster from themoviedb site 
# sometimes fetching poster may give network errors dure to connection issue so we can simply remove it and it will work fine with texts
def fetch_poster(movie_id):    
    url = 'https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US'.format(movie_id)
  
    data = requests.get(url)
    data = data.json()
    print(data)
    poster_path = data['poster_path']

    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


# Function to recommend the movie and posters
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key = lambda x: x[1])[1:6]
    
    recommended_movie = []
    recommended_movie_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        
        recommended_movie.append(movies.iloc[i[0]].title)
        # fetch poster from api= d3692ce4fecfce91e8bcc1676841d69d
        recommended_movie_posters.append(fetch_poster(movie_id))
    return recommended_movie,recommended_movie_posters

# ____________________________________________________________________________________________________________________________________________________________________



    




#__________________________________________________________________________GENRE BASED________________________________________________________________________________________

# function to recommend the movie based on genres 

def build_chart(genre, percentile=0.85):
    # movie_index = gen_movies[gen_movies['genre'] == genre].index[0]
    df = gen_movies[gen_movies['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False)[1:6]
    st.text(qualified.head(5))
    recommended_movie = ['1','2','3','4','5']
    
    
    
    return recommended_movie

# _______________________________________________________________________________________________________________________________________________________________________







# _____LOADING DATA FROM PKL_______________________________________________________________________


movies_dict = pickle.load(open('movies_dict.pkl','rb'))
rmovies_dict = pickle.load(open('emovies.pkl','rb'))
gen_movies_dict = pickle.load(open('gen_movies.pkl','rb'))
c_ratings_dict = pickle.load(open('c_ratings.pkl','rb'))
rmovies = pd.DataFrame(rmovies_dict)
movies= pd.DataFrame(movies_dict)
gen_movies = pd.DataFrame(gen_movies_dict)
c_ratings = pd.DataFrame(c_ratings_dict)
similarity = pickle.load(open('similarity.pkl','rb'))
# ___________________________________________________________________________________________________







# _____WEBSITE INTERFACE STARTED_________________________________________________________________________________________________

# setting up the page configerations of the web interface
st.set_page_config(
    page_title="RECOMM",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Added a siderBar to the page telling the all types of recommendations that this site provides
with st.sidebar:
    selected = option_menu(
        menu_title = "Movies",
        options = ["Genre Based","Top Selected","To your Choice"]

    )


# Creating logo,title, and some informations about the application using st.image,title,header and write
st.image("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAH4AAAB+CAMAAADV/VW6AAAAZlBMVEXzUyWBvAYFpvD/ugjz8/MAovB7uQD19Pf49vvz9vb59vPz9vvz+fvzqZq815j/tgCazPH51przSAvz0svb58rK4/L158sAnvDzpZbd6vP07d725cXz4t7m7NzzOwDY5cTz6unu8evcvsQ4AAAA+ElEQVRoge3ZyQ6CMBhFYVSwFByZxAHU939JLRjr0lxNasw5+58vadjdKA5aBB+atwsp625NKmU8b8t2KdSenN+tpCrz5BdtXwj15Z0357nUKvX8soiEioGv5hOhr/FneHh4eHh4eHh4eHh4eHh4eHh4eHh4eHj4f+fDTkn2VEpd3PG1kur8kBZbreE2NlLj7U+MqKH5TOnxic8ff19L5e722kgd/K+XbXaJ0K6+P4A5rqW2+QufTIWSgW/WMyF4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHj4t/mgU1Jcb6T27vawlWqs57UZcdwRTS41DsC/MaLCh+gGpDbqQSgyiyUAAAAASUVORK5CYII=", width=100)
st.title('RECOMM')
st.header('A quick overview of the whole system')
st.write(' This recommendation system is divided into three parts, the one Ranked list which recommend the movies on the basis of the ratings ratings and popularity of a particuler movie,One using genres Based filtering ,collbarative filtering and last one which recommends the movie on the basis of similarities between the movie and popularity among people')


# creating a selectbox to tak the input from the user and provide recommendations
recommend_top_movie = st.selectbox('TOP MOVIES',rmovies['title'].values)


selected_genre_movie = st.selectbox('SELECT GENRE',gen_movies['genre'].values)

# creating response for all the UI action
if st.button('Recommend_gen'):
    recommended_movie_names =  build_chart(selected_genre_movie)
    col1,col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(recommended_movie_names[1])
        # st.image(recommended_movie_posters[1])

    with col2:
        st.text(recommended_movie_names[1])
        # st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        # st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        # st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])



# making search place
selected_movie = st.selectbox('TO YOUR CHOICE?',movies['title'].values)



if st.button('Recommend'):
    recommended_movie_names,recommended_movie_posters =  recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])
 
