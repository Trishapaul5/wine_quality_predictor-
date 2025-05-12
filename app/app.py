import gradio as gr
import sys
import os

# Add the project root directory to sys.path (app/ is directly in the project root)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
print("Project root:", project_root)
print("sys.path:", sys.path)

from src.models.predict_model import load_model, predict_wine_quality

# Define the feature columns expected by the model
feature_columns = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

# Load the classification model
print("Loading classification model...")
cls_model = load_model()
print("Model loaded successfully.")

def predict(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol):
    try:
        wine_features = {
            'fixed acidity': fixed_acidity,
            'volatile acidity': volatile_acidity,
            'citric acid': citric_acid,
            'residual sugar': residual_sugar,
            'chlorides': chlorides,
            'free sulfur dioxide': free_sulfur_dioxide,
            'total sulfur dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': pH,
            'sulphates': sulphates,
            'alcohol': alcohol
        }
        print("Input features to predict:", wine_features)  # Debug
        quality_class = predict_wine_quality(wine_features, cls_model, feature_columns)
        return f"Quality Classification: {quality_class}"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Create the Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Wine Quality Predictor")
    gr.Markdown("Input the wine features below to predict its quality classification (Good or Not as Good).")
    
    with gr.Row():
        with gr.Column():
            fixed_acidity = gr.Slider(4, 16, value=7.5, label="Fixed Acidity")
            volatile_acidity = gr.Slider(0, 2, value=0.5, label="Volatile Acidity")
            citric_acid = gr.Slider(0, 1, value=0.25, label="Citric Acid")
            residual_sugar = gr.Slider(0, 16, value=2.5, label="Residual Sugar")
            chlorides = gr.Slider(0, 0.7, value=0.08, label="Chlorides")
            free_sulfur_dioxide = gr.Slider(0, 80, value=15.0, label="Free Sulfur Dioxide")
        with gr.Column():
            total_sulfur_dioxide = gr.Slider(0, 300, value=45.0, label="Total Sulfur Dioxide")
            density = gr.Slider(0.98, 1.01, value=0.996, label="Density")
            pH = gr.Slider(2.5, 4.5, value=3.3, label="pH")
            sulphates = gr.Slider(0, 2, value=0.65, label="Sulphates")
            alcohol = gr.Slider(8, 15, value=10.5, label="Alcohol")
            submit_button = gr.Button("Predict")
    
    quality_output = gr.Textbox(label="Quality Classification")
    
    submit_button.click(
        fn=predict,
        inputs=[
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol
        ],
        outputs=quality_output
    )

if __name__ == "__main__":
    iface.launch()