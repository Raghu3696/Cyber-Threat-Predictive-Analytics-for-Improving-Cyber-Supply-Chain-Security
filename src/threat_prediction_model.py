from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_threat_model(df):
    """
    Train a basic ML model to predict high-risk cyber threats
    """
    X = df[["risk_score"]]
    y = df["high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test
