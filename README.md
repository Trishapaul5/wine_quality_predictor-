🍷 Wine Quality Predictor: Sip the Magic of Machine Learning! 🍷
Welcome to the Wine Quality Predictor—a delightful blend of data science and wine tasting! This project uses a Random Forest model to predict whether a wine is "Good" or "Not as Good" based on its chemical properties. With an interactive Gradio interface, you can play sommelier and explore wine quality like never before. Ready to uncork some insights? Let’s dive in! 🥂

🌟 What’s This Project About?
Imagine you’re at a vineyard, swirling a glass of wine, trying to guess its quality. Is it a fine vintage or just okay? This project does the guessing for you—using machine learning! We’ve trained a Random Forest Classifier to analyze 11 chemical features (like acidity, alcohol, and sulphates) and classify wines into "Good" or "Not as Good." The best part? You can test it yourself with a sleek Gradio web app—no wine expertise required! 🍾
Project Highlights

Task: Binary classification of wine quality ("Good" or "Not as Good").
Model: Random Forest Classifier (with a test accuracy of 0.9345—pretty impressive, right?).
Interface: A user-friendly Gradio app to input wine features and get instant predictions.
Dataset: The Wine Quality Dataset from Kaggle, fetched via kagglehub.


📂 Project Structure
Here’s a peek at how this winery is organized:
wine_quality_predictor/
├── app/                    # 🍇 The heart of the tasting room
│   ├── app.py              # Launches the Gradio interface
│   └── data.py             # Handles data prep and model training
├── src/                    # 🛠️ The winemaker’s toolkit
│   ├── data/              # Data handling scripts
│   ├── features/          # Feature engineering magic
│   ├── models/            # Model training and prediction logic
│   └── visualization/     # Visual insights (because who doesn’t love a good chart?)
├── requirements.txt        # 📜 Ingredients list for this recipe
├── README.md               # 📖 The guide you’re reading now
└── .gitignore              # Keeps the cellar clean (ignores data/ and models/)

Note: The data/ and models/ directories are excluded via .gitignore since they contain generated files. You’ll create them when you run the project!

🚀 Get Started: Setup Instructions
Ready to taste some machine learning magic? Follow these steps to set up the project on your local machine.
Prerequisites

Python 3.8+ (because great wine needs a great vintage 🕰️)
Git (to clone this repository)

Step-by-Step Setup

Clone the Repository:
git clone https://github.com/Trishapaul5/wine_quality_predictor.git
cd wine_quality_predictor/wine_quality_predictor


Install Dependencies:Pour in the ingredients listed in requirements.txt:
pip install -r requirements.txt


Prepare the Data & Train the Model:Let’s get the wine ready and train our sommelier (the model):
python app/data.py


This script downloads the dataset using kagglehub, preprocesses it, trains the Random Forest model, and saves it to models/wine_quality_classification_model.pkl.


Launch the Gradio Interface:Open the tasting room and start predicting:
python app/app.py


A URL (e.g., http://127.0.0.1:7861) will appear—open it in your browser to access the Gradio app.




🎉 How to Use the App
The Gradio interface is your virtual wine tasting table! Here’s how to use it:

Adjust the Sliders:

Input the chemical properties of your wine, like fixed acidity, volatile acidity, alcohol, etc. Don’t worry if you’re not a chemist—the sliders make it easy! 🧪
Example: Set fixed acidity to 7.5, alcohol to 10.5, and so on.


Click "Predict":

Hit the "Predict" button, and voila! The app will tell you if the wine is "Good" or "Not as Good."


Sip & Reflect:

Imagine sipping that wine while marveling at the power of machine learning. Cheers to data science! 🥂




📊 Under the Hood: The Data & Model
Dataset
We’re using the Wine Quality Dataset from Kaggle—a collection of chemical properties and quality ratings for various wines. It’s downloaded automatically via kagglehub when you run app/data.py.
Model

Algorithm: Random Forest Classifier 🌳
Features: 11 chemical properties (e.g., pH, sulphates, alcohol).
Performance: Achieved a test accuracy of 0.9345—our model knows its wines! 🍷

The model is trained using a scikit-learn pipeline that includes preprocessing (e.g., StandardScaler) and classification. The trained model is saved as wine_quality_classification_model.pkl for predictions.

🌱 What’s Next?
This project is just the first sip—there’s more to explore! Here are some ideas to enhance the winery:

Unit Tests: Add tests to ensure the pipeline and model are as reliable as a fine vintage.
Hugging Face Spaces: Deploy the Gradio app on Hugging Face Spaces for the world to taste! 🌍
Feature Engineering: Experiment with new features or models (e.g., XGBoost) to boost performance.
Visualizations: Add more charts to src/visualization/ to showcase insights from the data.


📜 License
This project is licensed under the MIT License—feel free to share, modify, and sip to your heart’s content! 🍷

🙌 Let’s Connect!
I’d love to hear your thoughts on this project! If you have suggestions, questions, or just want to chat about wine and data science, reach out:

GitHub: Trishapaul5

Cheers to building something amazing together! 🥂
