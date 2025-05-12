import pandas as pd
import joblib
import os

def load_model():
    """Load the trained classification model."""
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    model_path = os.path.join(base_path, 'models', 'wine_quality_classification_model.pkl')
    print(f"Attempting to load model from: {model_path}")
    try:
        cls_model = joblib.load(model_path)
        print("Model loaded successfully.")
        print("Pipeline steps:", cls_model.steps)
        return cls_model
    except Exception as e:
        raise FileNotFoundError(f"Failed to load model: {str(e)}")

def predict_wine_quality(wine_features_dict, cls_model, feature_columns):
    """
    Predict wine quality class given its features.

    Args:
        wine_features_dict (dict): Dictionary of wine features (e.g., {'fixed acidity': 7.5, ...}).
        cls_model: The trained classification model (scikit-learn Pipeline).
        feature_columns (list): List of expected feature column names.

    Returns:
        str: Predicted quality class ("Good" or "Not as Good").
    """
    try:
        # Convert the dictionary to a DataFrame with one row
        sample = pd.DataFrame([wine_features_dict])
        print("Input features DataFrame:")
        print(sample)

        # Ensure all expected columns are present, fill missing with 0
        for col in feature_columns:
            if col not in sample.columns:
                sample[col] = 0.0

        # Reorder columns to match the training data
        sample = sample[feature_columns]

        # Add the 'Id' column to match the model's expected input
        sample['Id'] = 0  # Default value; Id is not used in prediction
        print("DataFrame after adding Id column:")
        print(sample)

        # Ensure all columns are float64 to match training data (except Id, which can be int)
        for col in sample.columns:
            if col != 'Id':
                sample[col] = sample[col].astype('float64')
        print("Prepared DataFrame for prediction:")
        print(sample)
        print("Data types:")
        print(sample.dtypes)

        # Inspect the preprocessor's expected columns
        try:
            preprocessor = cls_model.named_steps['preprocessor']
            print("Preprocessor transformers:", preprocessor.transformers_)
        except Exception as e:
            print(f"Could not inspect preprocessor: {str(e)}")

        # Make prediction using the model pipeline
        prediction = cls_model.predict(sample)
        print("Raw prediction:", prediction)

        # Convert prediction to quality class (0 -> "Not as Good", 1 -> "Good")
        quality_class = "Good" if prediction[0] == 1 else "Not as Good"
        print(f"Predicted quality class: {quality_class}")
        return quality_class

    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")