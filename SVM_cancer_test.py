from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=2)
# metadata 
print(breast_cancer_wisconsin_diagnostic.metadata) 
  
# variable information 
print(breast_cancer_wisconsin_diagnostic.variables) 


num_folds = 5

# Define the cross-validation object
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)


svc = SVC(C=1, kernel='linear')
svc.fit(X, y)


# The  c
#   argument allows us to specify the cost of a violation to the margin. When the  c
#   argument is small, then the margins will be wide and many support vectors will be
# on the margin or will violate the margin. When the  c
#   argument is large, then the margins will be narrow and there will be few support vectors on the margin or violating the margin.
tuned_parameters = [{'C': [0.001, 0.01, 0.1, 1, 5, 10, 100]}]

# GridSearchCV()
#   to perform cross-validation.
#   In order to use this function, we pass in relevant information about the set of
#   models that are under consideration. The following command indicates that we want 
#   perform 10-fold cross-validation to compare SVMs with a linear kernel, using a 
#   range of values of the cost parameter:

clf = GridSearchCV(SVC(kernel='linear'), tuned_parameters, cv=10, scoring='accuracy')
clf.fit(X, y)
print(clf.best_params_)
print(confusion_matrix(y_test, clf.best_estimator_.predict(X_test)))


from sklearn.metrics import auc
from sklearn.metrics import roc_curve

svc2 = SVC(C=1, kernel='linear')
svc2.fit(X_train, y_train)

y_train_score = svc2.decision_function(X_train)
false_pos_rate, true_pos_rate, _ = roc_curve(y_train, y_train_score,pos_label=1)
roc_auc = auc(false_pos_rate, true_pos_rate)

import matplotlib.pyplot as plt
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(14,6))
ax1.plot(false_pos_rate, true_pos_rate, label='SVM $\gamma = 1$ ROC curve (area = %0.2f)' % roc_auc, color='b')
ax1.set_title('Training Data')
for ax in fig.axes:
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")

plt.show()

print("fin")