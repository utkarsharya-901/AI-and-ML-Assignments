import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 1. Sample Movie Dataset (Simulating MovieLens / TMDB dataset)
def get_sample_data():
    data = {
        'movie_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'title': [
            'The Dark Knight',
            'Inception',
            'Interstellar',
            'The Matrix',
            'Pulp Fiction',
            'The Godfather',
            'Fight Club',
            'Forrest Gump',
            'The Shawshank Redemption',
            'Avatar'
        ],
        'genres': [
            'Action Crime Drama',
            'Action Adventure Sci-Fi',
            'Adventure Drama Sci-Fi',
            'Action Sci-Fi',
            'Crime Drama',
            'Crime Drama',
            'Drama',
            'Drama Romance',
            'Drama',
            'Action Adventure Fantasy Sci-Fi'
        ],
        'overview': [
            'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.',
            'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.',
            'A team of explorers travel through a wormhole in space in an attempt to ensure humanity survival across distant stars and galaxies.',
            'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.',
            'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
            'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
            'An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into much more.',
            'The presidencies of Kennedy and Johnson, the Vietnam War, and other historical events unfold from the perspective of an Alabama man with an IQ of 75.',
            'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
            'A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home.'
        ]
    }
    return pd.DataFrame(data)

# 2. Movie Recommender Class
class MovieRecommender:
    def __init__(self, df):
        self.df = df
        # Combine genres and overview into a single content feature
        self.df['content'] = self.df['genres'] + " " + self.df['overview']
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['content'])
        # Compute cosine similarity matrix
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        # Construct a reverse map of indices and movie titles
        self.indices = pd.Series(self.df.index, index=self.df['title'].str.lower()).drop_duplicates()

    def get_recommendations(self, title, top_n=3):
        title_clean = title.lower().strip()
        if title_clean not in self.indices:
            return f"Movie '{title}' not found in database."
        
        idx = self.indices[title_clean]
        
        # Get the pairwise similarity scores of all movies with that movie
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the scores of the N most similar movies (ignoring the input movie itself)
        sim_scores = sim_scores[1:top_n+1]
        
        movie_indices = [i[0] for i in sim_scores]
        
        # Return top N most similar movies with their similarity score
        result = self.df.iloc[movie_indices][['title', 'genres']].copy()
        result['similarity_score'] = [round(score[1], 4) for score in sim_scores]
        return result

def main():
    print("1. Initializing Movie Recommendation System...")
    df = get_sample_data()
    print(f"Loaded database with {len(df)} movies.\n")
    
    recommender = MovieRecommender(df)
    
    test_movie = "Inception"
    print(f"2. Generating Top 3 Recommendations for: '{test_movie}'")
    print("=" * 60)
    recommendations = recommender.get_recommendations(test_movie, top_n=3)
    print(recommendations.to_string(index=False))
    print("=" * 60)
    
    test_movie_2 = "The Godfather"
    print(f"\n3. Generating Top 3 Recommendations for: '{test_movie_2}'")
    print("=" * 60)
    recommendations_2 = recommender.get_recommendations(test_movie_2, top_n=3)
    print(recommendations_2.to_string(index=False))
    print("=" * 60)

if __name__ == '__main__':
    main()
