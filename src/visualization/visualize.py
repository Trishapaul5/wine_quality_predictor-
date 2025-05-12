import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

def plot_classification_results(y_cls_test, y_pred_cls, save_path="../data/processed"):
    """
    Plot confusion matrix and print classification report.
    
    Args:
        y_cls_test (pd.Series): True labels.
        y_pred_cls (np.ndarray): Predicted labels.
        save_path (str): Directory to save visualizations.
    """
    print("\nClassification Report:")
    print(classification_report(y_cls_test, y_pred_cls))

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_cls_test, y_pred_cls)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_path, 'wine_quality_classification_results.png'))
    plt.close()