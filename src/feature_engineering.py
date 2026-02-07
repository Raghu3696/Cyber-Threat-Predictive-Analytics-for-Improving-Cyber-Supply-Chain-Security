def extract_features(df):
    """
    Generate simple risk-based features
    """
    df["high_risk"] = df["risk_score"] > 0.6
    return df
