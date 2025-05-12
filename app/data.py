import warnings
import numpy as np
import sys
import os

# Add the project root directory to sys.path (app/ is directly in the project root)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
print("Project root:", project_root)  # Debug
print("sys.path:", sys.path)  # Debug

from src.data.preprocess import load_data, exploratory_data_analysis, preprocess_data
from src.features.build_features import prepare_data, create_preprocessor
from src.models.train_model import train_classification_models
from src.visualization.visualize import plot_classification_results

def main():
    """Run the end-to-end wine quality prediction pipeline for classification."""
    warnings.filterwarnings('ignore')
    np.random.seed(42)

    # Define directory paths relative to the project root
    data_raw_dir = os.path.join(project_root, 'data', 'raw')
    data_processed_dir = os.path.join(project_root, 'data', 'processed')
    models_dir = os.path.join(project_root, 'models')

    print("Creating directories...")
    os.makedirs(data_raw_dir, exist_ok=True)
    os.makedirs(data_processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    print("Directories created.")

    try:
        print("Step 1: Loading data...")
        wine_data = load_data()
        print("Data loaded successfully.")

        print("Step 2: Saving raw data...")
        raw_data_path = os.path.join(data_raw_dir, 'wine_quality_raw.csv')
        wine_data.to_csv(raw_data_path, index=False)
        print(f"Raw data saved as '{raw_data_path}'")

        print("Step 3: Performing exploratory data analysis...")
        exploratory_data_analysis(wine_data, save_path=data_processed_dir)
        print("EDA completed.")

        print("Step 4: Preprocessing data...")
        wine_data = preprocess_data(wine_data)
        print("Data preprocessed successfully.")

        print("Step 5: Saving preprocessed data...")
        preprocessed_data_path = os.path.join(data_processed_dir, 'processed_wine_data.csv')
        wine_data.to_csv(preprocessed_data_path, index=False)
        print(f"Preprocessed data saved as '{preprocessed_data_path}'")

        print("Step 6: Preparing data for training...")
        X, X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = prepare_data(wine_data)
        print("Data prepared successfully.")

        print("Step 7: Creating preprocessor...")
        preprocessor = create_preprocessor(X)
        print("Preprocessor created.")

        print("Step 8: Training classification models...")
        cls_model = train_classification_models(X_train, y_cls_train, X_test, y_cls_test, preprocessor)
        print("Classification models trained successfully.")

        print("Step 9: Making predictions for visualization...")
        y_pred_cls = cls_model.predict(X_test)
        print("Predictions made.")

        print("Step 10: Plotting classification results...")
        plot_classification_results(y_cls_test, y_pred_cls, save_path=data_processed_dir)
        print("Pipeline completed successfully.")

    except Exception as e:
        print(f"Error in pipeline execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()