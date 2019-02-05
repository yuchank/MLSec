import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Read in the data from the CSV file
df = pd.read_csv('./ch2/payment_fraud.csv')
print(df.sample(3))

# Convert categorical feature into dummy variables with one-hot encoding
df = pd.get_dummies(df, columns=['paymentMethod'])
print(df.sample(3))

# Split data set into training and test sets
X_train, X_test, y_train, y_test = \
    train_test_split(df.drop('label', axis=1), df['label'], test_size=0.33, random_state=17)

# Initialize and train classifier model
# clf = LogisticRegression(solver='lbfgs')
clf = LogisticRegression(solver='liblinear')
clf.fit(X_train, y_train)

# Make predictions on test set
y_pred = clf.predict(X_test)

# Compare test set predictions with ground truth labels
print(accuracy_score(y_pred, y_test))
print(confusion_matrix(y_test, y_pred))
