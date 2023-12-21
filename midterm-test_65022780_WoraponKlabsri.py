from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

FILE_PATH = 'data/'
FILE_NAME = 'car_data.csv'

df = pd.read_csv(FILE_PATH + FILE_NAME)

df.drop(columns=['User ID'], inplace=True)
df = df.fillna(method='pad')
encoders = []

for i in range(1, len(df.columns)):
    enc = LabelEncoder()
    df.iloc[:, i] = enc.fit_transform(df.iloc[:, i])
    encoders.append(enc)
    
x = df.iloc[:, 1:4]
y = df['Gender']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

print('Train:')
display(x_train)
print('\nTest:')
display(x_test)

model = DecisionTreeClassifier(criterion='entropy')
model.fit(x, y)

x_pred = [14, 20000, 'Yes']

for i in range(1, len(df.columns) - 1):
    x_pred[i] = encoders[i].transform([x_pred[i]])
    
x_pred_adj = np.array(x_pred).reshape(1, 3)

y_pred = model.predict(x_pred_adj)
print('\nPrediction:', y_pred[0])
score = model.score(x, y)
print('Accuracy:', '{:.2f}\n'.format(score))

feature = x.columns.tolist()
Data_class = y.tolist()

plt.figure(figsize = (75,50))
_ = plot_tree(model,
              feature_names = feature,
              class_names = Data_class,
              label = "all",
              impurity = True,
              precision = 3,
              filled = True,
              rounded = True,
              fontsize = 16)

plt.show()

feature_importances = model.feature_importances_
feature_names = ['Age', 'Annual_Salary', 'Purchased']
sns.set(rc={'figure.figsize':(11.7, 8.27)})
sns.barplot(x = feature_importances, y = feature_names)