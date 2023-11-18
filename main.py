from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = FastAPI()


# CORS (Cross-Origin Resource Sharing) settings
origins = [
    "http://127.0.0.1:8000",
    "http://localhost:5173",
    "https://workshala-navy.vercel.app",
    "https://course2.onrender.com",
    "http://localhost:5000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


df = pd.read_csv("all_courses.csv")
df = df.drop_duplicates()
df['Level'].fillna(df['Level'].mode()[0], inplace=True)
df['Skills Covered'].fillna(df['Skills Covered'].mode()[0], inplace=True)
df = df[['Title', 'Level', 'Description', 'Skills Covered']]
df['tags'] = df['Description'] + ' ' + df['Title'] + ' ' + df['Skills Covered']
df['tags'] = df['tags'].str.lower()
df.dropna(subset=['Title'], inplace=True)
df['Level'].fillna('Unknown', inplace=True)
df['tags'] = df['tags'].str.replace('[^\w\s]', " ")

new_df = df[['Title', 'tags']]
new_df['tags'].fillna(' ', inplace=True)

text_data = new_df['tags']

# Count Vectorizer
vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(text_data)
cosine_sim = cosine_similarity(matrix, matrix)


# Function to get recommendations based on user input
def get_recommendations(tags, top_n=5):
    user_vector = vectorizer.transform([tags])
    cosine_sim_with_selected_tags = cosine_similarity(user_vector, matrix)

    sim_scores = list(enumerate(cosine_sim_with_selected_tags[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    indices = [score[0] for score in sim_scores[:top_n]]
    recommendations = [
        {"courses": df['Title'].iloc[i],
         "Description": df["Description"].iloc[i],
         "Level": df["Level"].iloc[i],
         "Skills Covered": df['Skills Covered'].iloc[i]}
        for i in indices
    ]
    return recommendations

# FastAPI endpoint for recommendations
@app.get("/recommend/{tags}")
async def get_recommendations_endpoint(tags: str, top_n: int = 5):
    try:
        recommendations = get_recommendations(tags, top_n)
        return {"message": "Recommendations for tags", "courses": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during recommendation generation: {str(e)}")
