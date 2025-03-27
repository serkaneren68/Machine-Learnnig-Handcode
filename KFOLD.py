from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

# Load your dataset here
X = # Your dataset
y = # Your target variable

# Define the number of folds
num_folds = 10

# Define the cross-validation object
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Define your model here
model = LinearRegression()

# Perform k-fold cross-validation
scores = cross_val_score(model, X, y, cv=kfold)

# Print the cross-validation scores
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(np.mean(scores)))