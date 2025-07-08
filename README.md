# YouTube Sentiment Insights (MLOps Project)

This project is an end-to-end MLOps pipeline that trains a sentiment analysis model and deploys it as an API. A companion Chrome extension then uses this API to analyze the sentiment of comments on any YouTube video in real-time.

## Features

-   **MLOps Pipeline:** Uses DVC to create a reproducible ML pipeline for data ingestion, preprocessing, training, and evaluation.
-   **Experiment Tracking:** Integrates with MLflow to log experiments, track model performance, and register the best models.
-   **Sentiment Analysis API:** A FastAPI server that exposes the trained model for sentiment prediction.
-   **YouTube Chrome Extension:** A browser extension that fetches comments from a YouTube video, calls the API for sentiment analysis, and displays an aggregated summary with charts and word clouds.

## Architecture

The project is composed of two main parts: the MLOps Pipeline and the Frontend Application.

1.  **MLOps Pipeline (`src/`, `dvc.yaml`)**:
    -   **`src/`**: Contains the production-ready source code for the pipeline stages.
    -   **`dvc.yaml`**: Defines the DVC pipeline stages (data ingestion, preprocessing, model building, etc.).
    -   **`params.yaml`**: Contains hyperparameters and configuration for the pipeline and MLflow.
    -   **`main.py`**: A script to run the entire DVC pipeline sequentially.

2.  **Application & Frontend (`app.py`, `yt-chrome-plugin-frontend/`)**:
    -   **`app.py`**: A FastAPI application that loads the trained model and vectorizer to serve prediction requests.
    -   **`yt-chrome-plugin-frontend/`**: The source code for the Chrome extension that provides the user interface.

---

## Setup and Installation

Follow these steps to set up and run the project locally.

### Prerequisites

-   Python >= 3.12
-   Git
-   A running MLflow tracking server.

### 1. Clone the Repository

```bash
git clone https://github.com/bachnguyen0175/YouTube-Sentiment-Insights-MLOps.git
cd YouTube-Sentiment-Insights-MLOps
```

### 2. Backend & Pipeline Setup

First, set up the Python environment and install the required dependencies.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure MLflow and API Keys

Before running the pipeline or the application, you need to configure your MLflow server URI and your YouTube Data API key.

-   **MLflow**: Open `params.yaml` and set the `mlflow_uri` to your MLflow tracking server's address.
-   **YouTube API Key**:
    1.  Navigate to the `yt-chrome-plugin-frontend/` directory.
    2.  Make a copy of `config.js.template` and rename it to `config.js`.
    3.  Open `config.js` and replace `'YOUR_YOUTUBE_API_KEY_HERE'` with your actual YouTube Data API key.

---

## How to Run

### 1. Run the ML Pipeline

To train the model, run the DVC pipeline. This command will execute all the stages defined in `dvc.yaml`, from data ingestion to model evaluation.

```bash
dvc repro
```

This will generate the trained model (`model/lgbm_model.pkl`) and the TF-IDF vectorizer (`model/tfidf_vectorizer.pkl`).

### 2. Start the Sentiment Analysis API

Once the model is trained, start the FastAPI server.

```bash
python app.py
```

The API will be available at `http://localhost:8000`. You can see the documentation at `http://localhost:8000/docs`.

### 3. Load the Chrome Extension

Finally, load the unpacked extension into your Chrome browser.

1.  Open Chrome and navigate to `chrome://extensions/`.
2.  Enable **"Developer mode"** using the toggle in the top-right corner.
3.  Click on the **"Load unpacked"** button.
4.  Select the `yt-chrome-plugin-frontend` directory from this project.

Once loaded, you can navigate to any YouTube video page and click the extension's icon in your browser toolbar to see the sentiment analysis of the comments.