import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Assuming 'cyber_threats.csv' is your dataset
data = pd.read_csv('cyber_threats.csv')


# Fill missing values
data.fillna(method='ffill', inplace=True)


data = pd.get_dummies(data)


X = data.drop('threat_type', axis=1)
y = data['threat_type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

#show predictions
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.show()

