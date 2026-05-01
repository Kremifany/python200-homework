import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
num_samples = 500
years_experience = np.random.randint(2, 21, size=num_samples)
slope = (200_000 - 60_000) / 18
intercept = 60_000
salaries = intercept + slope * years_experience + np.random.normal(0, 10_000, size=num_samples)

data = {'Years_of_Experience': years_experience, 'Salary': salaries}
df = pd.DataFrame(data)
print(df.describe())

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Years_of_Experience', y='Salary', data=df, color='blue', label='Data Points')
sns.regplot(x='Years_of_Experience', y='Salary', data=df, scatter=False, color='red', label='Regression Line')
plt.title('Years of Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()  

X= df[['Years_of_Experience']]
y = df['Salary']    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr= LinearRegression()
lr.fit(X_train, y_train)
lr_score_train = lr.score(X_train, y_train)
print(f"Training Set R^2 Score: {lr_score_train:.2f}")    
lr_score_test = lr.score(X_test, y_test)
print(f"Test Set R^2 Score: {lr_score_test:.2f}")

y_pred = lr.predict(X_test)
mean_absolute_error_value = mean_absolute_error(y_test, y_pred)
mean_squared_error_value = mean_squared_error(y_test, y_pred)
r2_score_value = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mean_absolute_error_value:.2f}")
print(f"Mean Squared Error: {mean_squared_error_value:.2f}")
print(f"R^2 Score: {r2_score_value:.2f}")
# lr.coef_ means the slope of the regression line, which indicates how much 
# the salary is expected to increase for each additional year of experience.
# lr.intercept_ represents the y-intercept of the regression line, which is the expected salary when years of experience is zero. In this case, it should be close to 60,000, which is the base salary for someone with no experience in our synthetic dataset.
print(f"Slope (Coefficient): {lr.coef_[0]:.2f}")
print(f"Intercept: {lr.intercept_:.2f}")
coofficients =  lr.coef_
intercept = lr.intercept_
X = np.linspace(0, 20, 100)
y = coofficients * X + intercept

plt.scatter(X, y, color='blue', label= f'y = {coofficients[0]:.2f}x + {intercept:.2f}')

plt.title('Linear Regression Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.grid()
plt.show()