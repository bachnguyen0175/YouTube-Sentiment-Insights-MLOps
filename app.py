from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from pathlib import Path
from typing import List

# Import the reusable functions from your common utilities
from src.utils.common import preprocess_text, load_model, load_object

app = FastAPI(
    title="Sentiment Analysis API",
    description="An API to predict the sentiment of a given comment.",
    version="1.0.0"
)

# --- CORS Middleware ---
# This allows the frontend (Chrome extension) to communicate with this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- API Data Models ---
class CommentRequest(BaseModel):
    comment: str

class PredictionResponse(BaseModel):
    comment: str
    sentiment: int

class TimestampedComment(BaseModel):
    text: str
    timestamp: str

class TimestampedCommentRequest(BaseModel):
    comments: List[TimestampedComment]

class TimestampedPredictionResponse(BaseModel):
    comment: str
    sentiment: int
    timestamp: str

# --- Model & Vectorizer Loading ---
# Load the model and vectorizer once at startup to be efficient
model_path = Path("./model/lgbm_model.pkl")
vectorizer_path = Path("./model/tfidf_vectorizer.pkl")

model = load_model(model_path)
vectorizer = load_object(vectorizer_path)

# --- API Endpoints ---
@app.get("/")
def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"message": "Welcome to the Sentiment Analysis API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: CommentRequest):
    """
    Predicts the sentiment of a single comment.
    - **comment**: The text of the comment to analyze.
    """
    # 1. Preprocess the comment using the existing utility function
    preprocessed_comment = preprocess_text(request.comment)
    
    # 2. Transform the comment using the loaded vectorizer
    transformed_comment = vectorizer.transform([preprocessed_comment])
    
    # 3. Convert the sparse matrix to a dense format for the model
    dense_comment = transformed_comment.toarray()
    
    # 4. Make a prediction using the loaded model
    prediction = model.predict(dense_comment)[0]
    
    # 5. Return the response in the specified format
    return PredictionResponse(comment=request.comment, sentiment=int(prediction))

@app.post("/predict_with_timestamps", response_model=List[TimestampedPredictionResponse])
def predict_with_timestamps(request: TimestampedCommentRequest):
    """
    Predicts the sentiment for a batch of comments with timestamps.
    - **comments**: A list of objects, each with 'text' and 'timestamp'.
    """
    if not request.comments:
        return []

    # 1. Extract texts and timestamps from the request
    texts = [item.text for item in request.comments]
    timestamps = [item.timestamp for item in request.comments]

    # 2. Preprocess all texts in a batch
    preprocessed_texts = [preprocess_text(text) for text in texts]
    
    # 3. Transform the texts using the loaded vectorizer
    transformed_texts = vectorizer.transform(preprocessed_texts)
    
    # 4. Convert the sparse matrix to a dense format
    dense_texts = transformed_texts.toarray()
    
    # 5. Make predictions for the entire batch
    predictions = model.predict(dense_texts)
    
    # 6. Combine the results and return the response
    response = [
        TimestampedPredictionResponse(
            comment=text, 
            sentiment=int(pred), 
            timestamp=ts
        ) for text, pred, ts in zip(texts, predictions, timestamps)
    ]
    
    return response

# --- Main execution ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
