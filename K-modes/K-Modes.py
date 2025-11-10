import pandas as pd
from matplotlib import pyplot as plt
from kmodes.kmodes import KModes




def define_number_of_clusters(categorical_features,n):
    sse = []

    for k in range(2, n):
       kmodes = KModes(n_clusters=k, init='Huang', n_init=10, max_iter=500, verbose=1)
       kmodes.fit_predict(categorical_features)
       sse.append(kmodes.cost_)

    print("Displaying plot to apply the Elbow method")
    plt.plot(range(2, n), sse)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()


# This function provides us the top 2 movies by rating for each cluster
def top_movies(group):
    return group.nlargest(2, 'aver_rating')



def propose_movie(user, ratings, movies):

    # 1. We use the ratings of the specific user
    user_ratings1 = ratings[ratings['userId'] == user]

    # 2. Merge user's ratings with movie information
    user_movie_ratings = pd.merge(user_ratings1, movies, on='movieId')

    # 3. Select relevant columns

    user_ratings_df = user_movie_ratings[['userId', 'movieId', 'rating', 'cluster']]
    # 4. Calculate the average rating per cluster for the user
    cluster_mean_ratings = user_ratings_df.groupby('cluster')['rating'].mean().reset_index()

    # 5. Create a data frame with the low rating cluster
    low_rating_clusters = cluster_mean_ratings[cluster_mean_ratings['rating'] < 3.5]['cluster']

    # 6. Create a data frame with the columns movieId, title, cluster, aver_rating that includes all the movies
    #    except the movies in the low rating clusters by the user and the movies that the user has already seen
    movies_for_user = movies[~movies['cluster'].isin(low_rating_clusters)]
    movies_for_user = movies_for_user[["movieId", "title", "cluster", "aver_rating"]]
    movies_for_user = movies_for_user[~movies_for_user['movieId'].isin(user_ratings_df)]

    if movies_for_user.empty:
        print("Sorry, there are no movies available for this user in clusters with a mean rating below 3.5.")
    else:
        # Group movies by cluster and apply the top_movies function
        top_movies_by_cluster = movies_for_user.groupby('cluster').apply(top_movies)

        print("You may also like the following movies", end="")
        for title in top_movies_by_cluster["title"]:
            print("=" * 30)
            print(title)





# Read csv file
movies = pd.read_csv("movies.csv", sep=",", header=0)
ratings = pd.read_csv("ratings.csv", sep=",", header=0)


## In the dataset movies we have to find the missing values and remove them
print("\nWhich columns have missing values? Should print True for movies")
print(movies.isnull().any())
# Dropping rows with NaN values
print("\nRemove rows that have at least one NaN value in any column, and store the result in another variable named movies")
movies = movies.dropna(axis=0, how='any')


##Also in the dataset ratings we have to find the missing values and take care of them
print("\nWhich columns have missing values? Should print True for ratings")
print(ratings.isnull().any())
# Dropping rows with NaN values
print("\nRemove rows that have at least one NaN value in any column, and store the result in another variable named ratings")
ratings = ratings.dropna(axis=0, how='any')


#We want to determine the number of clusters that minimize the sse.
#We 'll apply the embow rule to make our decision.
#For this, we have to use the categorical features of data frame movies.)
movies_categorical_features = movies.iloc[:,2:]
maxNumberOfClusters = 40
define_number_of_clusters(movies_categorical_features, maxNumberOfClusters)


# Execute KModes again with k=20 to get the final clusters.
km = KModes(n_clusters=20, init='Huang', n_init=10, verbose=1)
clusters = km.fit_predict(movies_categorical_features)
movies['cluster'] = clusters



# Merge movie data with ratings data
merged_df = pd.merge(movies, ratings, on='movieId')

# Group merged DataFrame by movie title and calculate mean rating for each movie.
average_ratings = merged_df.groupby('title')['rating'].mean().reset_index()
average_ratings = average_ratings.rename(columns={'rating': 'aver_rating'})
# Merge average ratings with movie DataFrame
movies = pd.merge(movies, average_ratings, on='title', how='left')

user = int(input("Give me the name of the user that you are interested in..."))
movies.apply(propose_movie(user,ratings,movies))








