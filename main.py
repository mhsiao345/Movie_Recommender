import pandas as pd
from tkinter import *
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_csv('TMDb_updated.CSV')

# make a copy
df = data.copy()

# drop all movie entries that does not have a written overview
df.dropna(inplace=True)

# drop all movie with 'plot unknown' in overview
df = df[df['overview'] != 'Plot unknown.']

# sort dataset by highest vote counts
df = df.sort_values(['vote_count', 'vote_average'], ascending=False)

# create a temp dataframe
df_temp = df.title

# drop any duplicate titles keeping only the 1st entry
df_temp.drop_duplicates(keep='first', inplace=True)

# merge the temp dataframe
df = pd.concat([df, df_temp], axis=1)

# drop all duplicate movie keeping only the one with the highest vote
df.dropna(inplace=True)

# reset the dataframe index
df.reset_index(inplace=True)

# remove extra columns that were inserted during the data clean-up process
df = df.iloc[:, 2:-1]

# create instance to remove all stop words
tfidf_vector = TfidfVectorizer(stop_words='english')

# fit and transform the data
tfidf_matrix = tfidf_vector.fit_transform(df['overview'])

# construct similarity matrix
similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)


# -------------------------------- REVIEW ----------------------------------- #
def get_review():
    movie_input = entry_movie.get().lower()

# checks against movie titles and displays rating and review if found
    try:
        review = df[df.title.str.lower() == movie_input]['overview'].values[0]
        rating = df[df.title.str.lower() == movie_input]['vote_average'].values[0]
        messagebox.showinfo(title='Movie Review', message=f'Rating {rating} ★\'s\n\n{review}')

# if not found, display error
    except IndexError:
        messagebox.showwarning(title='Not Found!',
                               message=f'Movie name misspelled or not in database. Re-enter.')


# -------------------------------- SEARCH ----------------------------------- #
def search_movie():
    movie_input = entry_movie.get().lower()

# checks against movie titles and gets similarity scores if found
    try:
        row = df.index[df.title.str.lower() == movie_input]
        movie_similar_scores = list(enumerate(similarity_matrix[row[0]]))
        movie_similar_scores.sort(key=lambda x: x[1], reverse=True)

# match the top ten similarity scores to row index and outputs the title along with the vote average
        recommended = ''
        for item in movie_similar_scores[1:11]:
            recommended += 'Rating ' + str(df['vote_average'].iloc[item[0]]) + ' ★\'s  '
            recommended += str(df['title'].iloc[item[0]]) + '\n'
        messagebox.showinfo(title='Suggested List of Similar Movies!', message=recommended)

# if not found, display error
    except IndexError:
        messagebox.showwarning(title='Not Found!',
                               message=f'Movie name misspelled or not in database. Re-enter.')


# ---------------------------- UI SETUP ------------------------------- #
window = Tk()
window.title('!Da Movie Recommender Tool! 😁')
window.config(pady=30, bg='#EAF6F6')

main_img = PhotoImage(file='movies.png')
canvas = Canvas(width=500, height=280, bg='#EAF6F6', highlightthickness=0)
canvas.create_image(250, 140, image=main_img)
canvas.grid(column=0, row=1, columnspan=2)

welcome = Label(text='Da Movie Recommender Tool', font=('Arial', 18, 'bold'), fg='green', highlightthickness=0)
welcome.grid(column=0, row=0, columnspan=2)
info = Label(text='Enter a movie and get it\'s review or search for\n'
                  'similar flicks from a database of 9,000+ movies', font=('Arial', 12, 'bold'), highlightthickness=0)
info.grid(column=0, row=2, columnspan=2)

movie_name = Label(text='Enter movie name:', font=('Arial', 12, 'bold'), fg='green', highlightthickness=0)
movie_name.grid(column=0, row=3, columnspan=2)

entry_movie = Entry(width=30)
entry_movie.focus()
entry_movie.grid(column=0, row=4, columnspan=2, ipady=2, pady=8)

get_review = Button(text='Get review', width=13, font=('Arial', 12, 'bold'), command=get_review)
get_review.grid(column=0, row=5)

search_button = Button(text='Search', width=13, font=('Arial', 12, 'bold'), command=search_movie)
search_button.grid(column=1, row=5)

window.mainloop()


'''
credits to:

Behic Guven (https://www.youtube.com/watch?v=YG0Jet6WNQ4) - whose guide I used to 
vectorize the reviews and perform matrix scoring.

Sankha Subhra Mondal (https://www.kaggle.com/sankha1998/tmdb-top-10000-popular-movies-dataset/code) - who
made available this TMDb_updated.CSV (Version 55, downloaded 12APR2021) dataset on kaggle.
'''
