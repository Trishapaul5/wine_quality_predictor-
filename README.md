Wine Quality Predictor
This project predicts the quality of wine (classified as "Good" or "Not as Good") based on its chemical properties using a Random Forest classification model. The project includes data preprocessing, model training, and a Gradio web interface for interactive predictions.
Project Structure
wine_quality_predictor/
├── app/
│   ├── app.py              # Gradio interface for predictions
│   └── data.py             # Data loading, preprocessing, and model training
├── src/
│   ├── data/              # Data handling scripts
│   ├── features/          # Feature engineering scripts
│   ├── models/            # Model training and prediction scripts
│   └── visualization/     # Visualization scripts
├── data/                  # Data directory (excluded in .gitignore)
├── models/                # Model directory (excluded in .gitignore)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation

Prerequisites

Python 3.8 or higher
Git

Setup Instructions

Clone the Repository:
git clone https://github.com/<your-username>/wine_quality_predictor.git
cd wine_quality_predictor/wine_quality_predictor


Install Dependencies:
pip install -r requirements.txt


Prepare the Data and Train the Model:

Run the data pipeline to download the dataset, preprocess it, and train the model:python app/data.py


This will download the wine quality dataset using kagglehub, preprocess it, train a Random Forest model, and save the model to models/wine_quality_classification_model.pkl.


Launch the Gradio Interface:

Start the web interface to make predictions:python app/app.py


Open the provided URL (e.g., http://127.0.0.1:7861) in your browser.
Use the sliders to input wine features and click "Predict" to see the quality classification.



Dataset
The project uses the Wine Quality Dataset from Kaggle, downloaded via kagglehub. The dataset contains chemical properties of wines and their quality ratings.
Model

Task: Binary classification ("Good" or "Not as Good") based on a quality threshold.
Algorithm: Random Forest Classifier.
Performance: Achieved a test accuracy of 0.9345 (as per the training output).

Usage

Adjust the sliders in the Gradio interface to input wine features (e.g., fixed acidity, volatile acidity, etc.).
Click "Predict" to see the quality classification.

Next Steps

Add unit tests for the data pipeline and model.
Deploy the Gradio app on Hugging Face Spaces.
Explore additional features or models for improved performance.

License
This project is licensed under the MIT License.
