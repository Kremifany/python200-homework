import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy.stats import pearsonr

import seaborn as sns   

# Pandas Q1

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)
print("\n---Pandas Q1---")
print(f"\nfirst three rows:\n{df.head(3)}")
print(f"\nthe shape is:{df.shape}")
print(f"\ndata types of each column:\n{df.dtypes}")


# Pandas Q2

print("\n---Pandas Q2---")
filtered_df = df[(df['grade'] > 80) & (df['passed'] == True)]
print(f"\nstudents who passed and have a grade above 80:\n{filtered_df}\n")


# Pandas Q3
print("\n---Pandas Q3---")
df['grade_curved'] = df['grade'] + 5
print(f"\nupdated DataFrame with curved grade:\n{df}\n")   

# Pandas Q4
print("\n---Pandas Q4---")
df["name_upper"] = df["name"].str.upper()
print(f"\nname and name_upper columns: \n{df[['name', 'name_upper']]}\n")

# Group the DataFrame by "city" and compute the mean grade for each city. Print the result.

print("\n---Pandas Q5---")  


mean_grade_by_city = df.groupby('city')['grade'].mean()
print(f"\nmean grade for each city:\n{mean_grade_by_city}\n")

print("\n---Pandas Q6---")

df['city'] = df['city'].replace('Austin', 'Houston')
print(f"\nname and city columns after replacement:\n{df[['name', 'city']]}\n")

print("\n---Pandas Q7---")

# Sort the DataFrame by "grade" in descending order and print the top 3 rows.
sorted_by_grade = df.sort_values(by='grade',ascending=False)
print(f"\ntop 3 rows of dataframe sorted by grade:\n{sorted_by_grade.head(3)}")

print("\n---NumPy Q1---")
# Create a 1D NumPy array from the list [10, 20, 30, 40, 50]. Print its shape, dtype, and ndim.
arr_1d = np.array([10, 20, 30, 40, 50])
print(f"\n1D array:\n{arr_1d}\n")
print(f"\nshape of the 1D array: {arr_1d.shape}")
print(f"\ndtype of the 1D array: {arr_1d.dtype}")
print(f"\nndim of the 1D array: {arr_1d.ndim}\n")   

print("\n---NumPy Q2---")

# Create the following 2D array and print its shape and size (total number of elements).
arr_2d = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(f"\narray:\n{arr_2d}\n")
print(f"\nshape of the array: {arr_2d.shape}")
print(f"\nsize of the array: {arr_2d.size}\n") 

print("\n---NumPy Q3---")
# Using the 2D array from Q2, slice out the top-left 2x2 block and print it.
# The expected result is [[1, 2], [4, 5]].
sliced_arr = arr_2d[:2,:2]
print(f"sliced array is:\n{sliced_arr}")

print("\n---NumPy Q4---")
# Create a 3x4 array of zeros using a built-in command. Then create a 2x5 array
# of ones using a built-in command. Print both.
zeros_array = np.zeros((3,4), dtype=int)
ones_array = np.ones((2,5), dtype=int)
print(f"zeros array:\n{zeros_array}\n")
print(f"ones array:\n{ones_array}\n")

print("\n---NumPy Q5---")
# Create an array using np.arange(0, 50, 5). First, think about what you expect 
# it to look like. Then, print the array, its shape, mean, sum, and standard deviation.
arr = np.arange(0, 50, 5)
print(f"array created with arrange (0,50,5):\n{arr}\n")
print(f"array shape:{arr.shape}\n array mean: {arr.mean()}\n array sum: {arr.sum()}\n array standart deviation: {np.std(arr)}\n")

print("\n---NumPy Q6---")
# Generate an array of 200 random values drawn from a normal distribution with mean 0
# and standard deviation 1 (use np.random.normal()).
# Print the mean and standard deviation of the result.
arr_rand = np.random.normal(0,1,200)
print(arr_rand)
print(f"mean: {arr_rand.mean()}")
print(f"standard deviation: {arr_rand.std()}")

print("\n---Matplotlib  Q1---")
# Plot the following data as a line plot. Add a title "Squares", x-axis label "x", and y-axis label "y".
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x,y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


print("\n---Matplotlib  Q2---")
# Create a bar plot for the following subject scores. Add a title "Subject Scores" and label both axes.
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]
plt.bar(subjects, scores, color="orange")
plt.title("Subject Scores")
plt.xlabel("Subject")
plt.ylabel("Scores")
plt.show()

print("\n---Matplotlib  Q3---")
# Plot the two datasets below as a scatter plot on the same figure.
# Use different colors for each, add a legend, and label both axes.

x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
plt.scatter(x1, y1, color="blue", label="dataset 1")
plt.scatter(x2, y2, color="red", label="dataset 2")
plt.title("Scatter plot of two datasets")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show() 

print("\n---Matplotlib  Q4---")
# Use plt.subplots() to create a figure with 1 row and 2 subplots side by side.
# In the left subplot, plot x vs y from Q1 as a line. In the right subplot,
# plot the subjects and scores from Q2 as a bar plot.
# Add a title to each subplot and call plt.tight_layout() before showing.
fig, ax = plt.subplots(1, 2)
ax[0].plot(x , y, color= 'blue',marker='o')
ax[0].set_title('Subplot 1: Line Plot')
ax[1].bar(subjects, scores, color='orange')
ax[1].set_title('Subplot 2: Bar plot')
plt.tight_layout()
plt.show()

print("\n---Descriptive Stats Question 1---")
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
# Given the list below, use NumPy to compute and print the mean, median,
# variance, and standard deviation. Label each printed value.
data_array = np.array(data)
print(f"data array: {data_array}")
print(f"mean: {data_array.mean()}")
print(f"median: {np.median(data_array)}")
print(f"variance: {data_array.var()}")
print(f"standard deviation: {data_array.std()}")

print("\n---Descriptive Stats Question 2---")
# Generate 500 random values from a normal distribution with mean 65
# and standard deviation 10 (use np.random.normal(65, 10, 500)).
# Plot a histogram with 20 bins.
# Add a title "Distribution of Scores" and label both axes.
scoresData = np.random.normal(65, 10, 500)
plt.hist(scoresData, bins=20, color="purple", edgecolor="black")
plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

print("\n---Descriptive Stats Question 3---")
# Create a boxplot comparing the two groups below. Label each box 
# ("Group A" and "Group B") and add a title "Score Comparison".
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.boxplot([group_a, group_b], tick_labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Score")
plt.show()

print("\n---Descriptive Stats Question 4---")
# You are given two datasets: one normally distributed and one 'exponential' distribution.
# Create side-by-side boxplots comparing the two distributions.
# Label each boxplot appropriately ("Normal" and "Exponential") and add a title "Distribution Comparison".
# Then, add a comment in your code briefly noting which distribution is more skewed, and which descriptive 
# statistic (mean or median) would provide a more appropriate measure of central tendency for each distribution.
normal_data = np.random.normal(50, 5, 200)  
skewed_data = np.random.exponential(10, 200)    
# skewed distribution is more skewed than normal distribution,
# skewed distribution: more appropriate measure of central tendency would be median,
# normal distribution: mean would be more appropriate measure of central tendency .

plt.boxplot([normal_data, skewed_data], tick_labels=["Normal", "Exponential"],showmeans=True)

plt.title("Distribution Comparison")
plt.ylabel("data")
plt.show()

print("\n---Descriptive Stats Question 5---")
# Print the mean, median, and mode of the following:``

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]
mean_val1 = np.mean(data1)
median_val1 = np.median(data1)
mode_val1 = stats.mode(data1,keepdims=True)


print(f"data1 Mean: {mean_val1}")    
print(f"data1 Median: {median_val1}")  
print(f"data1 Mode: {mode_val1.mode[0]}")  

mean_val2 = np.mean(data2)
median_val2 = np.median(data2)
mode_val2 = stats.mode(data2,keepdims=True)

print(f"data2 Mean: {mean_val2}")    
print(f"data2 Median: {median_val2}")  
print(f"data2 Mode: {mode_val2.mode[0]}")
# Why are the median and mean so different for data2? Add your answer as a comment in the code.
# the median and mean are so different because in data2 we have an outlier that is increasing the mean value.

print("\n---Hypothesis Question 1---")
# Run an independent samples t-test on the two groups below. Print the t-statistic and p-value.

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]
t_stat, p_val = stats.ttest_ind(group_a, group_b)
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_val:.10f}")

print("\n---Hypothesis Question 2---")
# using the p-value from Q1, write an if/else statement that prints
# whether the result is statistically significant at alpha = 0.05.

alpha = 0.05
if(p_val < alpha):
    print(f"Result is statistically significant at alpha = {alpha} and p-value={p_val:.10f}")
else:    
    print(f"Result is not statistically significant at alpha = {alpha} and p-value={p_val:.10f}")

print("\n---Hypothesis Question 3---")
# Run a paired t-test on the before/after scores below (the same students measured twice). Print the t-statistic and p-value.
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]
t_stat, p_val = stats.ttest_rel(before,after)
alpha = 0.05
if(p_val < alpha):
    print(f"Result is statistically significant at alpha = {alpha} and p-value={p_val:.10f}")
else:    
    print(f"Result is not statistically significant at alpha = {alpha} and p-value={p_val:.10f}")

print("\n---Hypothesis Question 4---")
# Run a one-sample t-test to check whether the mean of scores is significantly
# different from a national benchmark of 70. Print the t-statistic and p-value.
benchmark = 70
scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_stat, p_val = stats.ttest_1samp(scores,benchmark)
alpha = 0.05
if(p_val < alpha):
    print(f"Result is statistically significant at alpha = {alpha} and p-value={p_val:.10f}")
else:    
    print(f"Result is not statistically significant at alpha = {alpha} and p-value={p_val:.10f}")

print("\n---Hypothesis Question 5---")
# Re-run the test from Q1 as a one-tailed test to check whether group_a scores
# are less than group_b scores. Print the resulting p-value. Use the alternative parameter.

t_stat, p_val = stats.ttest_ind(group_b, group_a, alternative='greater')
print(f"P-value (one-tailed) = {p_val:.10f}")

print("\n---Hypothesis Question 6---")
# Write a plain-language conclusion for the result of Q1 (do not just say "reject the null hypothesis").
# Format it as a print() statement. Your conclusion should mention the direction of the difference
# and whether it is likely due to chance.
print("---Conclusion for Q1---")
print("- Null hypotesis: means of two groups are the same")
print("- Alternative hypothesis: A significant difference exists between the groups.")
print("- After a t-test: Since the p-value is extremely low,\n " \
" it is very unlikely that the difference is due to chance." \
"\n  Group B performed significantly better than Group A." \
"\n  from the one tailed test we can see that scores of group B are much higher than group A.")

print("\n---Correlation Question 1---")
# Compute the Pearson correlation between x and y below using np.corrcoef().
# Print the full correlation matrix, then print just the correlation coefficient (the value at position [0, 1]).
# What do you expect the correlation to be, and why? Add your answer as a comment in the code.

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
corr_matrix = np.corrcoef(x, y)
print("full correlation matrix:")
print(corr_matrix)
print(f"correlation coefficient: {corr_matrix[0, 1]}")
print ("I expect the correlation to be 1 because y is a function of x y=2x, which indicates a perfect positive correlation.")

print("\n---Correlation Question 2---")
# Use pearsonr() from scipy.stats to compute the correlation between x and y below.
# Print both the correlation coefficient and the p-value.


x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

corr_coeff, p_value = pearsonr(x, y)
print(f"Correlation coefficient: {corr_coeff}")
print(f"P-value: {p_value:.10f}")

print("\n---Correlation Question 3---")
# Create the following DataFrame and use df.corr()
# to compute the correlation matrix. Print the result.
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
corMatrix = df.corr()
print("Correlation matrix:")
print(corMatrix)    

print("\n---Correlation Question 4---")
# Create a scatter plot of x and y below, which have a negative relationship.
# Add a title "Negative Correlation" and label both axes.


x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]
plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

print("\n---Correlation Question 5---")
# Using the correlation matrix from Q3, create a heatmap with sns.heatmap().
# Pass annot=True so the correlation values appear in each cell, and add a title "Correlation Heatmap".

sns.heatmap(data=corMatrix, annot=True)
plt.title("Correlation Heatmap")
plt.show()


print("\n---Pipeline Question 1---")

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
print(f"Original array:\n{arr}\n")
def create_series(arr):
    series = pd.Series(arr, name="values")
    return series

def clean_data(series):
    cleaned_series = series.dropna()
    return cleaned_series

def summarize_data(series):
    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
    return summary

def data_pipeline(arr):
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary

result = data_pipeline(arr)
print("Summary of the data:")   
for key, value in result.items():
    print(f"{key}: {value}")

