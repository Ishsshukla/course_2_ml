# recommendation_module.py

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def create_similarity_matrix(df):
    df['tags'] = df['Description'] + ' ' + df['Title'] + ' ' + df['Skills Covered']
    df['tags'] = df['tags'].str.lower()
    df['tags'].fillna(' ', inplace=True)
    text_data = df['tags']

    # Count Vectorizer
    vectorizer = CountVectorizer()
    data_matrix = vectorizer.fit_transform(text_data)

    # Cosine similarity matrix
    similarity_matrix = cosine_similarity(data_matrix)
    return similarity_matrix

def search_courses(keyword, df):
    matching_courses = df[df['tags'].str.contains(keyword, case=False, na=False)]
    
    if not matching_courses.empty:
        return matching_courses['Title'].tolist()
    else:
        return []

def recommend_courses(keyword, similarity_matrix, df, top_n=5):
    #  index of the course matching the keyword
    keyword_index = df[df['tags'].str.contains(keyword, case=False, na=False)].index
    if not keyword_index.empty:
        keyword_index = keyword_index[0]
    else:
        return []

    #  similarity scores for the keyword course
    similarity_scores = list(enumerate(similarity_matrix[keyword_index]))

    # Sort based on similarity scores
    sorted_courses = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Exclude the keyword course itself
    sorted_courses = [index for index, _ in sorted_courses if index != keyword_index]

    # Return the top N recommended courses
    return df['Title'].iloc[sorted_courses[:top_n]].tolist()
