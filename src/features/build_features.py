from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def prepare_data(wine_data):
    """Prepare features and target variables, and split the data."""
    print("\n--- Data Preparation ---")
    # Define features and targets
    X = wine_data.drop(['quality', 'quality_binary'], axis=1)
    y_reg = wine_data['quality']  # For regression
    y_cls = wine_data['quality_binary']  # For classification

    # Split the data
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    return X, X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test

def create_preprocessor(X):
    """Create a preprocessing pipeline for the features."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.columns)
        ]
    )
    return preprocessor