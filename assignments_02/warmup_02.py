# # --- scikit-learn API ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

print("---scikit-learn Question 1 ---\n")
# Create a LinearRegression model, fit it to this data,
# and then predict the salary for someone with 4 years
# of experience and someone with 8 years.
# Print the slope (model.coef_[0]), the intercept (model.intercept_), and the two predictions. Label each printed value.

# salary = slope * years + intercept


# 1. Create the model instance
model = LinearRegression()

# 2. Fit the model to the data (this is where it calculates the slope and intercept)
model.fit(years, salary)

# 3. Predict salary for 4 and 8 years 
prediction_4 = model.predict([[4]])
prediction_8 = model.predict([[8]])

print(f"Predicted Salary for 4 years of experience: ${prediction_4[0]:,.2f}")
print(f"Predicted Salary for 8 years of experience: ${prediction_8[0]:,.2f}")
print(f"Model Slope: Coefficient: {model.coef_[0]:.2f}")
print(f"Model Intercept: ${model.intercept_:,.2f}")

print("---scikit-learn Question 2---\n")

x = np.array([10, 20, 30, 40, 50])

x2d = x.reshape(-1, 1)

print(f"x 2d array: {x2d.shape}")

# Why does scikit-learn need X to be 2D?
# scikit learn expect 2d array with columns of features 
# because of the "work" he will need to perfom later that requires matrix operations
# and different features that that 2d array will consist of
# the features could be more detailes that will come into 
# considiration for the result of the function like year,knowledge more different factors.

# scikit-learn Question 3
print("---scikit-learn Question 3---\n")
# K-Means is an unsupervised algorithm that follows the same create → fit → predict pattern as everything else in scikit-learn.
# Use the code below to generate a synthetic dataset with three natural clusters:

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
# Create a KMeans model with n_clusters=3 and random_state=42, fit it to X_clusters,
# and predict a cluster label for each point.
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)
print("kmeans.cluster_centers_:", kmeans.cluster_centers_)
print("Number of points in each cluster:", np.bincount(labels)) 

# Then create a scatter plot coloring each point by its cluster label,
# plot the cluster centers as black X's,
# add a title and axis labels.
# Save the figure to outputs/kmeans_clusters.png.   
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_clusters[:, 0], y=X_clusters[:, 1], hue=labels, palette='viridis', legend='full')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, c='black', label='Cluster Centers')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig('outputs/kmeans_clusters.png')
# Linear Regression
print("---Linear Regression---\n")
# The questions below all use the same synthetic medical costs dataset:
# 100 patients, each with an age (20 to 65), a smoker flag (0 = non-smoker, 1 = smoker),
# and an annual medical cost as the target. Generate it once and reuse the variables throughout.

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

print("Linear Regression Question 1\n")
plt.figure(figsize=(10, 6))
plt.scatter(age, cost,c=smoker,cmap="coolwarm")
plt.title('Medical Cost vs Age')
plt.xlabel('Age')
plt.ylabel('MedicalCost')


plt.savefig('outputs/cost_vs_age.png')
plt.show()
# I can see from the plot that smockers have higher costs of medicine than non smockers of the same age.
# also with the age the medicine cost grows
# Linear Regression Question 2
print("Linear Regression Question 2\n")
# Split the data into training and test sets using age as the only feature, an 80/20 split,
# and random_state=42. Reshape age to a 2D array before using it as X. Print the shapes of all four arrays.
x=age
y=cost
x2d = age.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(x2d, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}\n")  

print("Linear Regression Question 3\n")
# Fit a LinearRegression model to your training data from Question 2.
# Print the slope and intercept. Then predict on the test set and print:
# RMSE: np.sqrt(np.mean((y_pred - y_test) ** 2))
# R² on the test set: model.score(X_test, y_test)
# Add a comment interpreting the slope in plain English -- what does it mean for medical costs?
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
print(f"Model Slope: Coefficient: {model_lr.coef_[0]:.2f}")
print(f"Model Intercept: ${model_lr.intercept_:,.2f}")
y_pred = model_lr.predict(X_test)
RMSE = np.sqrt(np.mean((y_pred - y_test) ** 2))
R2 = model_lr.score(X_test, y_test)
print(f"Root Mean Squared Error: {RMSE:.2f}")  
print(f"R^2 Score: {R2:.2f}")

# the slope means that medical cost grows with age

print("Linear Regression Question 4\n")
# Now add smoker as a second feature and fit a new model.
X_full = np.column_stack([age, smoker])
# Split, fit, and print the test R². Compare it to the R² from Question 3 
# -- does adding the smoker flag help? Print both coefficients:
X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(X_full, y, test_size=0.2, random_state=42)
print(f"X_full_train shape: {X_full_train.shape}")
print(f"X_full_test shape: {X_full_test.shape}")
print(f"y_full_train shape: {y_full_train.shape}")
print(f"y_full_test shape: {y_full_test.shape}\n")  

model_full = LinearRegression()
model_full.fit(X_full_train, y_full_train)
R2_full = model_full.score(X_full_test, y_full_test)
print(f"Test R^2 with age only: {R2:.2f}")
print(f"Test R^2 with age and smoker: {R2_full:.2f}")   
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])
# Add a comment interpreting the smoker coefficient: what does it represent in practical terms?
# Smoker coofficient adds more to the medical cost 14538 dollars for the same age of people

print("Linear Regression Question 5")
# A predicted vs actual plot is a standard tool for evaluating regression models.
# Each test observation becomes a dot: the model's prediction goes on the x-axis,the true value goes on the y-axis.
# A perfect model would place every point on the diagonal line where predicted equals actual.

# Using the two-feature model from Linear Regression Question 4,
# create this plot for the test set. Add a diagonal reference line, a title "Predicted vs Actual",
# labeled axes, and save to outputs/predicted_vs_actual.png.

y_pred_full = model_full.predict(X_full_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_full, y_full_test, color='blue', label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Prediction')
plt.title('Predicted vs Actual')
plt.xlabel('Predicted Medical Cost')
plt.ylabel('Actual Medical Cost')
plt.legend()
plt.grid()

plt.savefig('outputs/predicted_vs_actual.png')
plt.show()
# Add a comment: what does it mean when a point falls above the diagonal? What about below?
# A point above the diagonal means that the model predicted a higher result than the actual value (diagonal)
# A point below the diagonal means that the model predicted a lower result than the actual value (diagonal)
