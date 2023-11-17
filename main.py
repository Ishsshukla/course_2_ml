from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# CORS (Cross-Origin Resource Sharing) settings
origins = ["http://127.0.0.1:8000/", "http://localhost:5173", "https://courseapi-s0hm.onrender.com/", "http://localhost:5000"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Load your DataFrame and create similarity matrix
# (Note: I'm assuming you have the necessary functions in the recommendation_module and data_loader modules.)
from recommendation_module import create_similarity_matrix
from data_loader import load_dataframe

df = load_dataframe()
similarity_matrix = create_similarity_matrix(df)

# Count Vectorizer for course tags
vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(df['tags'].fillna(''))

# Function to get recommendations based on user input
def get_recommendations(tags, top_n=5):
    user_vector = vectorizer.transform([tags])
    cosine_sim_with_selected_tags = cosine_similarity(user_vector, matrix)

    sim_scores = list(enumerate(cosine_sim_with_selected_tags[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    indices = [score[0] for score in sim_scores[:top_n]]
    recommendations = [
        {"courses": df['title'].iloc[i],
         "Description": df["Description"].iloc[i],
         "Level": df["Level"].iloc[i],
         "Duration": df["Duration"].iloc[i],
         "Skills Covered": df['Skills Covered'].iloc[i],
         "prerequisites": df['prerequisites'].iloc[i],
         "URL": df['URL'].iloc[i]}
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
