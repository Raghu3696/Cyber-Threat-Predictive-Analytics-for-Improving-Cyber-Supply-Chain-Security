from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained threat prediction model
    """
    predictions = model.predict(X_test)

    return {
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": classification_report(y_test, predictions)
    }
