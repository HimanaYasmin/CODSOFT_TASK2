# Data handling
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")
df.head()
df.info()
df.shape
df.columnsdf.isnull().sum()
df = df.dropna()
df.isnull().sum()
df['Year'] = df['Year'].str.replace('(', '', regex=False)
df['Year'] = df['Year'].str.replace(')', '', regex=False)
df['Year'] = df['Year'].astype(int)
df['Duration'] = df['Duration'].str.replace(' min', '', regex=False)
df['Duration'] = df['Duration'].astype(int)
df['Votes'] = df['Votes'].str.replace(',', '', regex=False)
df['Votes'] = df['Votes'].astype(int)
df['Rating'] = df['Rating'].astype(float)
df.info()
df.head()
plt.figure(figsize=(8,5))
sns.histplot(df['Rating'], bins=10, kde=True)
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()
genre_rating = df.groupby('Genre')['Rating'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
genre_rating.plot(kind='bar')
plt.title("Top 10 Genres by Average Rating")
plt.xlabel("Genre")
plt.ylabel("Average Rating")
plt.show()
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['Votes'], y=df['Rating'])
plt.title("Votes vs Rating")
plt.xlabel("Votes")
plt.ylabel("Rating")
plt.show()
plt.figure(figsize=(6,4))
sns.heatmap(df[['Year','Duration','Votes','Rating']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
y = df['Rating']
X = df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Year', 'Duration', 'Votes']]
X.info()
le = LabelEncoder()

for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    X[col] = le.fit_transform(X[col])
X.head()
X.info()
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
X_train.shape, X_test.shape
y_train.shape, y_test.shape
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("Random Forest R2:", r2_score(y_test, rf_pred))
