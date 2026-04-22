# If you were loading this with pd.read_csv(), what parameter would you need to specify beyond the filename? 
# Write that observation as a comment at the top of your script before you write the load call.
#I would need to specify delimiter ";" because the data is separated by semicolons instead of commas.
# Load the dataset with the correct separator. Print the shape, the first five rows, and the data types of all columns.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("student_performance_math.csv", sep=';')
print(data.shape)
print(data.head())  
print(data.dtypes)

print("\nTask 1: Explore the Target Variable\n")
# Then plot a histogram of G3 with 21 bins (one per possible value, 0-20). 
# Add a title "Distribution of Final Math Grades", label both axes, and save to outputs/g3_distribution.png.
# You should see a cluster of zeros sitting apart from the main distribution.
# They represent the students who didn't take the final exam.

plt.figure(figsize=(10, 6))
sns.histplot(data['G3'], bins=21, kde=False, color='blue')
plt.title('Distribution of Final Math Grades')
plt.xlabel('Final Grade')
plt.ylabel('Number of Students') 
plt.savefig('outputs/g3_distribution.png')
plt.show()

print("\nTask 2: Preprocess the Data\n")

# Handle the G3=0 rows first. Filter them out and save the result to a new DataFrame.
data_filtered = data[data['G3'] != 0]
#  Print the shape before and after to confirm how many rows were removed.
print(f"Original dataset shape: {data.shape}")
print(f"Filtered dataset shape: {data_filtered.shape}")
print(f"Number of rows removed: {data.shape[0] - data_filtered.shape[0]}")
#  Add a comment explaining your reasoning -- why would keeping these rows distort the model?
# If we would keep those rows with g3=0 it would confuse the model that would think that g3=0 because the student did not studied enough

# Then convert the yes/no columns to 1/0 and the sex column to 0/1.
data_filtered['sex'] = data_filtered['sex'].map({'M': 1, 'F': 0})
yes_no_columns = ['schoolsup', 'internet', 'higher', 'activities']
for col in yes_no_columns:
    data_filtered[col] = data_filtered[col].map({'yes': 1, 'no': 0})
print(data_filtered.head())
# Now check something interesting before moving on.
# Compute the Pearson correlation between absences and G3 on both the original dataset and the filtered one,
# and print both values. The difference is striking. Add a comment explaining why filtering changes the result:
# what were students with G3=0 doing in the original data that made absences look like a weak predictor?
# You might want to explore scatter plots to help understand this.

correlation_original = data['absences'].corr(data['G3'])
correlation_filtered = data_filtered['absences'].corr(data_filtered['G3'])
print(f"Pearson correlation between absences and G3 in original dataset: {correlation_original:.2f}")
print(f"Pearson correlation between absences and G3 in filtered dataset: {correlation_filtered:.2f}")

# Why it looked like a "weak predictor":
# because when student were attending and got 0 in exam and when they were not attending they got 0 in exam
# so it was no strong correlation between those two
#######################################

# Pearson corelation visualization
# 1. Create a list of the labels and the values
labels = ['Original Data', 'Filtered Data (G3 > 0)']
correlations = [correlation_original, correlation_filtered]

plt.figure(figsize=(8, 6))

# Use distinct colors for positive vs negative
colors = ['firebrick' if val > 0 else 'seagreen' for val in correlations]
bars = plt.bar(labels, correlations, color=colors)

# Adjust the Y-limit so the 0.034 bar has space to show up
plt.ylim(-0.30, 0.10) 

# Add a prominent horizontal line at 0
plt.axhline(0, color='black', linewidth=1.5)

# Update the labels to show the jump
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (0.01 if yval > 0 else -0.02), 
             f'{yval:.3f}', ha='center', va='center', fontweight='bold')

plt.title('How Cleaning Data Changes the Story')
plt.ylabel('Pearson Correlation')
plt.savefig('outputs/pearson_correlation.png')
plt.show()

print("Task 3: Exploratory Data Analysis\n")

# Task 3: Exploratory Data Analysis

# 1.Compute the Pearson correlation between each numeric feature and G3 on the filtered dataset,
# and print them sorted from most negative to most positive.

numeric_data = data_filtered.select_dtypes(include=[np.number])
correlations_g3 = numeric_data.corr()['G3'].sort_values()

print("Pearson Correlation with G3 (Sorted):")
print(correlations_g3)

# Q: Are any results surprising?
# A: For me the most surprising was that the alcohol weekly and daily consumption has such a low negative correlation with failure at finals
# Another surprising finding was that absences not so much related to failures.
# the third finding that mother education more positive stronger then father education.

# Q: Which feature has the strongest relationship with G3?
# A: The strongest relationship is with G2 and G1 (previous grades), which is expected.
# Beyond previous grades, 'failures' has the strongest negative correlation.
# 'higher' (wanting to take higher education) shows a strong positive relationship.


# Visualization 1:
#  The Impact of Past Failures on Final Grades
# Since 'failures' is a key negative predictor, let's visualize the "ceiling" it creates.
plt.figure(figsize=(10, 6))
sns.boxplot(x='failures', y='G3', data=data_filtered, palette='Reds')
plt.title('Impact of Past Class Failures on Final Math Grade')
plt.xlabel('Number of Past Failures')
plt.ylabel('Final Grade (G3)')
 
# students with less previous failures has a higher success rate on finals

plt.savefig('outputs/failures_vs_g3_boxplot.png')
plt.show()


# Visualization 2: Does actually wanting to go to college (higher) change the grade outcome?
plt.figure(figsize=(10, 6))
# swarm plot sows single student as a dot
sns.swarmplot(x='higher', y='G3', data=data_filtered, hue='higher', palette='Set2', legend=False)
plt.title('The Ambition Gap: Students Wanting Higher Education vs. Grades')
plt.xlabel('Wants to go to University? (0 = No, 1 = Yes)')
plt.ylabel('Final Math Grade')


# Students that doesn't want to persue higher education dont have as high final grades
# as those that wish to go for higher education

plt.savefig('outputs/ambition_swarm.png')
plt.show()


# Visualization 3: Study Time vs. Social Life (Alcohol)
# Let's see if students who study more actually drink less, and how that affects G3.

plt.figure(figsize=(10, 6))

sns.pointplot(x='studytime', y='G3', hue='Walc', data=data_filtered, palette='autumn')
plt.title('Grades based on Study Time & Weekend Alcohol Use')
plt.xlabel('Study Time (1=Low to 4=High)')
plt.ylabel('Average Final Grade')

# Generally, more study time = better grades. 
# The lines of higher alcohol consumers (darker colors) inspite that they study a lot often dip or plateau
# if we compare to the lighter drinkers
# that suggest that all their hard working "undone" because of their drinking
plt.savefig('outputs/study_vs_alcohol_trends.png')
plt.show()

# Visualization 4: Correlation Heatmap
plt.figure(figsize=(12, 8))
# Focus on a subset of Influential Features
top_features = correlations_g3.index[-8:].tolist() + correlations_g3.index[:3].tolist()
sns.heatmap(data_filtered[top_features].corr(), annot=True, cmap='RdBu', fmt='.2f')
plt.title('Correlation Matrix: Top Influential Features')

# G1 and G2 are almost perfectly correlated with G3 (0.80-0.90), suggesting that 
# academic performance is highly consistent throughout the year. 
# 'failures' is negatively correlated with both study time and higher education goals.
plt.savefig('outputs/feature_correlation_heatmap.png')
plt.show()

# Task 4: Baseline Model
# Build the simplest possible model: use failures alone to predict G3.

X = data_filtered[["failures"]]
y = data_filtered["G3"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_failures = LinearRegression()
lr_failures.fit(X_train, y_train)
y_pred = lr_failures.predict(X_test)
RMSE = np.sqrt(np.mean((y_pred - y_test) ** 2))
R2 = lr_failures.score(X_test, y_test)

print("\nTask 4: Baseline — failures → G3 (LinearRegression)\n")
print(f"Slope (points of G3 per extra past failure): {lr_failures.coef_[0]:.4f}")
print(f"Intercept: {lr_failures.intercept_:.4f}")
print(f"RMSE on test set: {RMSE:.4f}")
print(f"R² on test set: {R2:.4f}")

# Slope is -1.43 that means that for every additional failure in the past grade is lower by 1.43
# RMSE is 3 which means that the forcast is off by 3 points which is noticeble 
# R² is low (~0.09): means that failures alone not changing so much G3
# In exploratory analysis we saw that the previous grades- G1 and G2 were the most powerful reason for growing G3

# Task 5: Build the Full Model

feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
X = data_filtered[feature_cols].values
y = data_filtered["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train, y_train)
y_pred_full = model_full.predict(X_test)
RMSE_full = np.sqrt(np.mean((y_pred_full - y_test) ** 2))
R2_train = model_full.score(X_train, y_train)
R2_test = model_full.score(X_test, y_test)

print("\nTask 5: Full model — Feature Guide features G3 \n")
print(f"Train R²: {R2_train:.4f}")
print(f"Test R²:  {R2_test:.4f}")
print(f"RMSE on test set: {RMSE_full:.4f}")
print(
    f"\nCompared to Task 4 baseline (test R² = {R2:.4f}), the full model adds "
    f"{R2_test - R2:+.4f} in test R² — more features help but not a lot  "
    f"because the most important features G1/G2 are not in this set"
)
print("\nCoefficients:")
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:12s}: {coef:+.3f}")

#the most surprising finding was that extra support does not help the grades to be higher
#train R2 = 0.17 and test R2 = 0.15 are very close that mens that there is no much overfitting
# low R2 saying that the model is too weak
# if you were deploying this model in production, which features would you keep and which would you drop?
# Justify your choices based on what you see in the numbers.

# I would consider to drop schoolsup beacause is misleading
# also drop activities, freetime and traveltime almost zero coofficient 
# I would keep features like: failures, higher, studytime, internet because
#  they affect the decision of the model in measurable way

# Task 6: Evaluate and Summarize
# A useful way to evaluate a regression model visually is a predicted vs actual plot. This is a scatter plot
# where each point in the test set becomes a dot, with the model's prediction (y_hat)
# on the x-axis and the true value (y) on the y-axis.
#  If the model were perfect, every point would fall exactly on the diagonal (predicted = actual).
#  Clusters or curves away from the diagonal reveal systematic errors that RMSE alone won't show you.
#  Random scattering around the diagonal is expected, and acceptable, prediction error.

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_full, y_test, alpha=0.65, edgecolors="none")
lo = min(float(np.min(y_pred_full)), float(np.min(y_test)))
hi = max(float(np.max(y_pred_full)), float(np.max(y_test)))
plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="predicted = actual")
plt.xlabel("Predicted G3 (test set)")
plt.ylabel("Actual G3 (test set)")
plt.title("Predicted vs Actual (Full Model)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("outputs/predicted_vs_actual.png")
plt.show()

# Q: Does the model struggle more at the high end, the low end, or is error roughly uniform?
# A: Not perfectly uniform—predictions bunch toward the middle, so the
#    extremes are harder: very high actual grades often appear above the diagonal (underpredicted),
#    and very low actual grades often below it (overpredicted). Mid-range grades sit closer to
#    the diagonal. RMSE alone does not show that pattern; the plot does.
# Q: What does a point above or below the diagonal mean? (x = predicted, y = actual)
# A: Above the diagonal → actual grade is higher than predicted (the model undershot). Below →
#    actual grade is lower than predicted (the model overshot).

# Summary:
# — After dropping G3=0, the filtered dataset has 357 rows; with an 80/20 split (random_state=42),
#   the test set has 71 students.

# RMSE off by 3 means that a prediction is off by 3 points
# test R2 is 0.15 means that the grade changes is not fully explained mostly beacase the previous grrades are not ion features set

#the bigger coofficient is internet meanes that the students that have internet likely will get better final grade
#the smallest coofficient is of scoolsup menas that even if the student gets support 
# not always helps to get better grade on finals. Maybe that happens because the weaker sudents get those schoolsup
# and they are sucseed in their own way but when we looking in all the students grades they look like not much of a success

# — One surprise: schoolsup’s strong negative weight even though “support” sounds like it should
#   raise grades.

print("Neglected Feature: The Power of G1")
# Neglected Feature: The Power of G1 — add first-period grade to the Task 5 feature set and refit.
feature_cols_with_g1 = feature_cols + ["G1"]
X_with_g1 = data_filtered[feature_cols_with_g1].values
y_with_g1 = data_filtered["G3"].values
X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_with_g1, y_with_g1, test_size=0.2, random_state=42
)

model_with_g1 = LinearRegression()
model_with_g1.fit(X_train_g1, y_train_g1)
R2_test_g1 = model_with_g1.score(X_test_g1, y_test_g1)

print(f"\nFull model with G1 added — test R²: {R2_test_g1:.4f}")
print("(Compared to Task 5 without G1, test R² jumps sharply once prior performance is included.)\n")

# High R2 with G1 in model doesn't mean that G1 causes G3.
# They both were caused by the same ability habits and engagement of student
# the model without G1/G2 very weak as the result have shown so there are no 
# predictions could be made for early intervention 
# If educators want to predict G1 they need more features to fit the model such as attendance, homework, 
# quizzes, some other ratings that will help with the prediction of the grades