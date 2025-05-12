import numpy as np
import os
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
import joblib

def train_regression_models(X_train, y_reg_train, X_test, y_reg_test, preprocessor):
    """
    Train and evaluate regression models.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_reg_train (pd.Series): Training target for regression.
        X_test (pd.DataFrame): Testing features.
        y_reg_test (pd.Series): Testing target for regression.
        preprocessor (ColumnTransformer): Preprocessing pipeline.
        
    Returns:
        Pipeline: The best regression model after training and tuning.
    """
    print("\n--- Training Regression Models ---")
    reg_models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'SVR': SVR(),
        'XGBoost': XGBRegressor(random_state=42)
    }

    reg_results = {}
    try:
        for name, model in reg_models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            cv_scores = cross_val_score(pipeline, X_train, y_reg_train, cv=5, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-cv_scores)

            pipeline.fit(X_train, y_reg_train)

            y_pred = pipeline.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
            test_r2 = r2_score(y_reg_test, y_pred)

            reg_results[name] = {
                'cv_rmse': rmse_scores.mean(),
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'model': pipeline
            }

            print(f"{name}:")
            print(f"  CV RMSE: {rmse_scores.mean():.4f}")
            print(f"  Test RMSE: {test_rmse:.4f}")
            print(f"  Test R²: {test_r2:.4f}")

        best_reg_model_name = min(reg_results, key=lambda x: reg_results[x]['test_rmse'])
        best_reg_model = reg_results[best_reg_model_name]['model']
        print(f"\nBest Regression Model: {best_reg_model_name}")
        print(f"Test RMSE: {reg_results[best_reg_model_name]['test_rmse']:.4f}")
        print(f"Test R²: {reg_results[best_reg_model_name]['test_r2']:.4f}")

        print("\n--- Hyperparameter Tuning (Regression) ---")
        if best_reg_model_name == 'Random Forest':
            print("Tuning Random Forest Regressor...")
            param_grid = {
                'model__n_estimators': [100, 200],
                'model__max_depth': [None, 10],
                'model__min_samples_split': [2, 5]
            }
            grid_search = GridSearchCV(best_reg_model, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_reg_train)
            best_reg_model = grid_search.best_estimator_

            y_pred = best_reg_model.predict(X_test)
            tuned_rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
            tuned_r2 = r2_score(y_reg_test, y_pred)

            print(f"Best Parameters: {grid_search.best_params_}")
            print(f"Tuned Test RMSE: {tuned_rmse:.4f}")
            print(f"Tuned Test R²: {tuned_r2:.4f}")
            print(f"Improvement in RMSE: {reg_results[best_reg_model_name]['test_rmse'] - tuned_rmse:.4f}")

        elif best_reg_model_name == 'XGBoost':
            print("Tuning XGBoost Regressor...")
            param_grid = {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.01, 0.1],
                'model__max_depth': [3, 5]
            }
            grid_search = GridSearchCV(best_reg_model, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_reg_train)
            best_reg_model = grid_search.best_estimator_

            y_pred = best_reg_model.predict(X_test)
            tuned_rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
            tuned_r2 = r2_score(y_reg_test, y_pred)

            print(f"Best Parameters: {grid_search.best_params_}")
            print(f"Tuned Test RMSE: {tuned_rmse:.4f}")
            print(f"Tuned Test R²: {tuned_r2:.4f}")
            print(f"Improvement in RMSE: {reg_results[best_reg_model_name]['test_rmse'] - tuned_rmse:.4f}")

        save_path = os.path.abspath('../models/wine_quality_regression_model.pkl')
        print(f"Attempting to save regression model to absolute path: {save_path}")
        try:
            joblib.dump(best_reg_model, save_path)
            print(f"Regression model saved as '{save_path}'")
        except Exception as e:
            print(f"Failed to save regression model: {str(e)}")
            raise

        return best_reg_model

    except Exception as e:
        print(f"Error during regression model training: {str(e)}")
        raise

def train_classification_models(X_train, y_cls_train, X_test, y_cls_test, preprocessor):
    """
    Train and evaluate classification models.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_cls_train (pd.Series): Training target for classification.
        X_test (pd.DataFrame): Testing features.
        y_cls_test (pd.Series): Testing target for classification.
        preprocessor (ColumnTransformer): Preprocessing pipeline.
        
    Returns:
        Pipeline: The best classification model after training and tuning.
    """
    print("\n--- Training Classification Models ---")
    cls_models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    cls_results = {}
    try:
        for name, model in cls_models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            cv_scores = cross_val_score(pipeline, X_train, y_cls_train, cv=5, scoring='accuracy')

            pipeline.fit(X_train, y_cls_train)

            y_pred = pipeline.predict(X_test)
            test_accuracy = accuracy_score(y_cls_test, y_pred)

            cls_results[name] = {
                'cv_accuracy': cv_scores.mean(),
                'test_accuracy': test_accuracy,
                'model': pipeline
            }

            print(f"{name}:")
            print(f"  CV Accuracy: {cv_scores.mean():.4f}")
            print(f"  Test Accuracy: {test_accuracy:.4f}")

        best_cls_model_name = max(cls_results, key=lambda x: cls_results[x]['test_accuracy'])
        best_cls_model = cls_results[best_cls_model_name]['model']
        print(f"\nBest Classification Model: {best_cls_model_name}")
        print(f"Test Accuracy: {cls_results[best_cls_model_name]['test_accuracy']:.4f}")

        print("\n--- Hyperparameter Tuning (Classification) ---")
        if best_cls_model_name == 'Random Forest':
            print("Tuning Random Forest Classifier...")
            param_grid = {
                'model__n_estimators': [100, 200],
                'model__max_depth': [None, 10],
                'model__min_samples_split': [2, 5]
            }
            grid_search = GridSearchCV(best_cls_model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_cls_train)
            best_cls_model = grid_search.best_estimator_

            y_pred = best_cls_model.predict(X_test)
            tuned_accuracy = accuracy_score(y_cls_test, y_pred)

            print(f"Best Parameters: {grid_search.best_params_}")
            print(f"Tuned Test Accuracy: {tuned_accuracy:.4f}")
            print(f"Improvement in Accuracy: {tuned_accuracy - cls_results[best_cls_model_name]['test_accuracy']:.4f}")

        elif best_cls_model_name == 'XGBoost':
            print("Tuning XGBoost Classifier...")
            param_grid = {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.01, 0.1],
                'model__max_depth': [3, 5]
            }
            grid_search = GridSearchCV(best_cls_model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_cls_train)
            best_cls_model = grid_search.best_estimator_

            y_pred = best_cls_model.predict(X_test)
            tuned_accuracy = accuracy_score(y_cls_test, y_pred)

            print(f"Best Parameters: {grid_search.best_params_}")
            print(f"Tuned Test Accuracy: {tuned_accuracy:.4f}")
            print(f"Improvement in Accuracy: {tuned_accuracy - cls_results[best_cls_model_name]['test_accuracy']:.4f}")

        # Save the best classification model
        save_path = os.path.abspath('../models/wine_quality_classification_model.pkl')
        print(f"Attempting to save classification model to absolute path: {save_path}")
        try:
            joblib.dump(best_cls_model, save_path)
            print(f"Classification model saved as '{save_path}'")
        except Exception as e:
            print(f"Failed to save classification model: {str(e)}")
            raise

        return best_cls_model

    except Exception as e:
        print(f"Error during classification model training: {str(e)}")
        raise
    
    