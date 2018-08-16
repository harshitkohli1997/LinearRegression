import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('customers.csv')

df.head()

sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=df)

X = df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
Y = df['Yearly Amount Spent']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

from sklearn.datasets import load_boston

boston = load_boston()

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)

from sklearn import metrics
metrics.mean_absolute_error(y_test,predictions)
metrics.mean_squared_error(y_test,predictions)