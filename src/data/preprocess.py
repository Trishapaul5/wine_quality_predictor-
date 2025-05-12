import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data():
    """
    Download and load the wine quality dataset from Kaggle or a fallback URL.
    
    Returns:
        pd.DataFrame: The loaded wine quality dataset.
    """
    try:
        print("Downloading wine quality dataset...")
        path = kagglehub.dataset_download("yasserh/wine-quality-dataset")
        print(f"Path to dataset files: {path}")

        print("Files in the downloaded directory:")
        for file in os.listdir(path):
            print(f" - {file}")

        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if csv_files:
            csv_path = os.path.join(path, csv_files[0])
            print(f"Found CSV file: {csv_path}")
            return pd.read_csv(csv_path)

        xlsx_files = [f for f in os.listdir(path) if f.endswith('.xlsx')]
        if xlsx_files:
            xlsx_path = os.path.join(path, xlsx_files[0])
            print(f"Found Excel file: {xlsx_path}")
            return pd.read_excel(xlsx_path)

        print("No suitable data files found. Falling back to URL...")
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/winequality-red.csv"
        return pd.read_csv(url, sep=';')

    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

def exploratory_data_analysis(wine_data, save_path="../data/processed"):
    """
    Perform exploratory data analysis and save visualizations.
    
    Args:
        wine_data (pd.DataFrame): The wine quality dataset.
        save_path (str): Directory to save visualizations.
    """
    print("\n--- Exploratory Data Analysis ---")
    print(f"Dataset Shape: {wine_data.shape}")
    print("\nFirst 5 rows:")
    print(wine_data.head())
    print("\nData Information:")
    print(wine_data.info())
    print("\nSummary Statistics:")
    print(wine_data.describe())
    print("\nMissing Values:")
    print(wine_data.isnull().sum())

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(wine_data['quality'], kde=True)
    plt.title('Distribution of Wine Quality')
    plt.xlabel('Quality')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    sns.boxplot(y=wine_data['quality'])
    plt.title('Boxplot of Wine Quality')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'wine_quality_distribution.png'))
    plt.close()

    plt.figure(figsize=(12, 10))
    correlation = wine_data.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', mask=mask)
    plt.title('Correlation Matrix of Wine Features')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'wine_correlation.png'))
    plt.close()

def preprocess_data(wine_data):
    """
    Preprocess the wine quality dataset.
    
    Args:
        wine_data (pd.DataFrame): The raw wine quality dataset.
        
    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    print("\n--- Feature Engineering ---")
    if 'type' in wine_data.columns:
        wine_data = pd.get_dummies(wine_data, columns=['type'], drop_first=True)

    wine_data['quality_binary'] = (wine_data['quality'] >= 7).astype(int)
    print(f"Binary quality distribution: {wine_data['quality_binary'].value_counts()}")

    return wine_data