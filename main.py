from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from recommendation_module import create_similarity_matrix, search_courses, recommend_courses
from data_loader import load_dataframe

app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing)
origins = [
    "http://workshala-in.vercel.app"
    "http://localhost:5173",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load DataFrame and create a similarity matrix
df = load_dataframe()
similarity_matrix = create_similarity_matrix(df)

@app.get("/recommend/{keyword}")
async def get_recommendations(keyword: str, top_n: int = 5):
   
    try:
        matching_courses = search_courses(keyword, df)
        if not matching_courses:
            return {"message": f"No matching courses found for {keyword}"}

        recommended_courses = recommend_courses(keyword, similarity_matrix, df, top_n=top_n)
        return {"message": f"Recommendations for {keyword}", "courses": recommended_courses}
    
    except Exception as e:
        return {"message": f"An error occurred during recommendation generation: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
