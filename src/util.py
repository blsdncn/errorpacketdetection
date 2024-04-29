# util.py
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, confusion_matrix, ConfusionMatrixDisplay

def load_model(model_path):
    """Load a model from a pickle file."""
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the given model using the test dataset."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        'Accuracy': accuracy_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions),
        'Recall': recall_score(y_test, predictions),
        'Log Loss': log_loss(y_test, probabilities) if probabilities is not None else None
    }
    return predictions, probabilities, metrics

def plot_confusion_matrix(y_test, predictions):
    """Plot the confusion matrix."""
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()

def save_model(model, model_path):
    """Save the model into a pickle file."""
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
