# Reddit Sentiment Analysis Pipeline

This project implements an end-to-end MLOps pipeline for sentiment analysis on Reddit comments. It uses DVC to version data and pipelines, and MLflow to track experiments and register models.

## Architecture

The project is structured as follows:

-   `src/`: Contains the production-ready source code for the pipeline.
    -   `components/`: Individual pipeline components for data ingestion, preprocessing, model training, evaluation, and registration.
    -   `pipeline/`: Orchestrates the execution of the components in stages.
    -   `utils/`: Common utility functions.
    -   `config/`: Configuration management.
    -   `entity/`: Defines custom data structures.
-   `research/`: Contains Jupyter notebooks used for experimentation and analysis.
-   `dvc.yaml`: Defines the DVC pipeline stages.
-   `params.yaml`: Contains hyperparameters and configuration for the pipeline.
-   `main.py`: Main script to run the full DVC pipeline.
-   `requirements.txt`: Project dependencies.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up MLflow:**
    -   Ensure you have an MLflow tracking server running.
    -   Update the `mlflow_uri` in `params.yaml` to point to your server.

## Running the Pipeline

To run the full pipeline, use DVC:

```bash
dvc repro
```

This command will execute all the stages defined in `dvc.yaml`, from data ingestion to model registration.

## Pipeline Stages

1.  **Data Ingestion:** Fetches the raw Reddit comment data.
2.  **Data Preprocessing:** Cleans and preprocesses the text data.
3.  **Model Building:** Trains a LightGBM model using TF-IDF features.
4.  **Model Evaluation:** Evaluates the model on the test set and logs metrics to MLflow.
5.  **Model Registration:** Registers the validated model in the MLflow Model Registry.
