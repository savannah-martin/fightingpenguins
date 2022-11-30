import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import _data
from sklearn.metrics import accuracy_score, precision_score, recall_score

#read in the data set
df = None 
path = 'penguin_data.csv'
df = pd.read_csv(path)

#drop unnecessary data
df = df.drop(columns=['date_egg', 'clutch_completion'], axis = 1)

#turn the strings into floats 
df['individual_id'] = LabelEncoder().fit_transform(df['individual_id'])
df['island'] = LabelEncoder().fit_transform(df['island'])
df['sex'] = LabelEncoder().fit_transform(df['sex'])
df['species'] = LabelEncoder().fit_transform(df['species'])

#print the dataset
df = df.dropna(axis=0)
print(df)


train_x, test_x, train_y, test_y= None,None,None,None

train_x, test_x, train_y, test_y = train_test_split(df.drop(['species'],axis=1),df['species'],test_size=0.2, train_size = 0.8, random_state= 42, shuffle=True )

scaler = None
original_train = train_x

scaler = StandardScaler()
scaler.fit(original_train)
original_train = scaler.transform(original_train)

support_vector_classifier = SVC(kernel='rbf')
support_vector_classifier.fit(original_train,train_y)

original_test = test_x
original_test = scaler.transform(original_test)


results_dict = {'Accuracy':0,'Precision':0,'Recall':0}

y_pred = support_vector_classifier.predict(test_x)
results_dict['Accuracy'] = accuracy_score(test_y, y_pred)
results_dict['Precision'] = precision_score(test_y, y_pred, average='micro')
results_dict['Recall'] = recall_score(test_y, y_pred, average='micro')

print(results_dict)

results_dict1 = {'Accuracy':0,'Precision':0,'Recall':0}

y_pred = support_vector_classifier.predict(test_x)
results_dict1['Accuracy'] = accuracy_score(test_y, y_pred)
results_dict1['Precision'] = precision_score(test_y, y_pred, average='macro')
results_dict1['Recall'] = recall_score(test_y, y_pred, average='macro')

print(results_dict1)


results_dict2 = {'Accuracy':0,'Precision':0,'Recall':0}

y_pred = support_vector_classifier.predict(test_x)
results_dict2['Accuracy'] = accuracy_score(test_y, y_pred)
results_dict2['Precision'] = precision_score(test_y, y_pred, average='weighted')
results_dict2['Recall'] = recall_score(test_y, y_pred, average='weighted')

print(results_dict2)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

# load plot helper code


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    scoring=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    if axes is None:
        _, axes = plt.subplots()

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes.legend(loc="best")
    return plt

svm = SVC(kernel='linear')
plot_learning_curve(svm, 'Support Vector Classifier', train_x, train_y)
plt.show()
