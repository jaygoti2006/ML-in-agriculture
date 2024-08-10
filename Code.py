
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv(r'C:\\Users\\acer1\\Downloads\\Poject\\Crop_recommendation.csv')


print(df.head())


numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()


features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']


Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.2, random_state=2)

# Initialize lists for storing model names and accuracies
acc = []
model = []


DecisionTree = DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5)
DecisionTree.fit(Xtrain, Ytrain)
predicted_values = DecisionTree.predict(Xtest)
acc.append(accuracy_score(Ytest, predicted_values))
model.append('Decision Tree')
print(f"Decision Tree's Accuracy is: {accuracy_score(Ytest, predicted_values) * 100:.2f}%")
print(classification_report(Ytest, predicted_values))

print(cross_val_score(DecisionTree, features, target, cv=5))

with open('DecisionTree.pkl', 'wb') as f:
    pickle.dump(DecisionTree, f)

NaiveBayes = GaussianNB()
NaiveBayes.fit(Xtrain, Ytrain)
predicted_values = NaiveBayes.predict(Xtest)
acc.append(accuracy_score(Ytest, predicted_values))
model.append('Naive Bayes')
print(f"Naive Bayes's Accuracy is: {accuracy_score(Ytest, predicted_values) * 100:.2f}%")
print(classification_report(Ytest, predicted_values))


print(cross_val_score(NaiveBayes, features, target, cv=5))


with open('NBClassifier.pkl', 'wb') as f:
    pickle.dump(NaiveBayes, f)


SVM = SVC(gamma='auto')
SVM.fit(Xtrain, Ytrain)
predicted_values = SVM.predict(Xtest)
acc.append(accuracy_score(Ytest, predicted_values))
model.append('SVM')
print(f"SVM's Accuracy is: {accuracy_score(Ytest, predicted_values) * 100:.2f}%")
print(classification_report(Ytest, predicted_values))


print(cross_val_score(SVM, features, target, cv=5))

LogReg = LogisticRegression(random_state=2)
LogReg.fit(Xtrain, Ytrain)
predicted_values = LogReg.predict(Xtest)
acc.append(accuracy_score(Ytest, predicted_values))
model.append('Logistic Regression')
print(f"Logistic Regression's Accuracy is: {accuracy_score(Ytest, predicted_values) * 100:.2f}%")
print(classification_report(Ytest, predicted_values))


print(cross_val_score(LogReg, features, target, cv=5))


with open('LogisticRegression.pkl', 'wb') as f:
    pickle.dump(LogReg, f)


RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain, Ytrain)
predicted_values = RF.predict(Xtest)
acc.append(accuracy_score(Ytest, predicted_values))
model.append('Random Forest')
print(f"Random Forest's Accuracy is: {accuracy_score(Ytest, predicted_values) * 100:.2f}%")
print(classification_report(Ytest, predicted_values))


print(cross_val_score(RF, features, target, cv=5))


with open('RandomForest.pkl', 'wb') as f:
    pickle.dump(RF, f)
