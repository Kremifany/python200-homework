import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import warnings
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.inspection import DecisionBoundaryDisplay

warnings.filterwarnings("ignore", category=RuntimeWarning)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
print(f"shape of iris dataset:\n{iris.data.shape}\n shape of iris target:\n{iris.target.shape}")

print("iris is a:", type(iris))                 # sklearn.utils.Bunch
print("X shape:", iris.data.shape)              # (150, 4)
print("y shape:", iris.target.shape)            # (150,)
print("feature names:", iris.feature_names)
print("target names:", iris.target_names)
print("iris is a:", type(iris))                 # sklearn.utils.Bunch
print("X shape:", iris.data.shape)              # (150, 4)
print("y shape:", iris.target.shape)            # (150,)
print("feature names:", iris.feature_names)
print("target names:", iris.target_names)

sns.countplot(x=y.map(dict(enumerate(iris.target_names))))
plt.title("Number of Flowers per Species")
plt.show()
sns.scatterplot(
    x=X["petal length (cm)"],
    y=X["petal width (cm)"],
    hue=y.map(dict(enumerate(iris.target_names)))
)
plt.title("Petal Length vs Petal Width")
plt.show()

# sns.pairplot(
#     pd.concat([X, y.rename("species")], axis=1),
#     hue="species"
# )
# plt.show()


# --- Preprocessing --- # Q1
print(X.head())
print("Preprocessing Question 1\n")
# Split X and y into training and test sets using an 80/20 split
# with stratify=y and random_state=42. Print the shapes of all four arrays.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# --- Preprocessing --- # Q2
print("\nPreprocessing Question 2")
# Fit a StandardScaler on X_train and use it to transform both X_train and X_test.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Q: why you fit the scaler on X_train only.
# A: we fit the scaler only on X_train because we want all statistics to come from what the model is allowed to see 
#  to avoid data leakage
X_test_scaled = scaler.transform(X_test)
# Print the mean of each column in X_train_scaled -- they should all be very close to 0.
print(X_train_scaled.mean(axis=0))
# --- KNN  --- # Q1
print("\nKNN Question 1")

# Build a KNeighborsClassifier with n_neighbors=5,
# fit it on the unscaled training data (X_train), and predict on the test set.
# Print the accuracy score and the full classification report.

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

preds = knn.predict(X_test)
knn_accuracy_score=accuracy_score(y_test, preds)
print("Accuracy:", knn_accuracy_score)
print("\nfull classification report:\n",classification_report(y_test, preds))

# --- KNN  --- # Q2
print("\nKNN Question 2")
# Repeat KNN Question 1 using the scaled data (X_train_scaled, X_test_scaled).
# Print the accuracy score. Add a comment: does scaling
# improve performance, hurt it, or make no difference?
# Why might that be for this particular dataset?
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

preds_onScaled = knn.predict(X_test_scaled)

print("preds_onScaled Accuracy:", accuracy_score(y_test, preds_onScaled))
print("\npreds_onScaled full classification report:\n",classification_report(y_test, preds_onScaled))
# Accuracy lowered in this dataset with the scaling was 1 became 0.93
# The accuracy changed because of the small dataset and neighbors have been changed

# --- KNN  --- # Q3
print("\nKNN Question 3\n")
# Using cross_val_score with cv=5, evaluate the k=5 KNN model on the unscaled training data.
# Print each fold score, the mean, and the standard deviation.
# Add a comment: is this result more or less trustworthy than a single train/test split, and why?

cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
print(cv_scores)           # accuracy on each fold
print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std:  {cv_scores.std():.3f}")
# Cross-validation is usually more trustworthy than a single train/test split because it averages
# performance over multiple folds

# --- KNN  --- # Q4
print("\nKNN Question 4")
# Loop over k values [1, 3, 5, 7, 9, 11, 13, 15].
# For each, compute 5-fold cross-validation accuracy on the unscaled training data
# and print k and the mean CV score. Add a comment identifying which k you would choose and why.
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
scores_by_k = {}
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    mean_score = scores.mean()
    std_score = scores.std()
    scores_by_k[k] = (mean_score, std_score)
    print(f"k={k:2d}:  mean={mean_score:.3f}  std={std_score:.3f}")

best_k = max(k_values, key=lambda k: (scores_by_k[k][0], -k))
print(f"Best k by mean CV: {best_k} (mean={scores_by_k[best_k][0]:.3f})")
# I would choose the k with the highest mean CV accuracy; if there's a tie, pick the smaller k.

# --- Classifier Evaluation ---  #Q1
print("\nClassifier Evaluation:\n")
# Using your predictions from KNN Question 1, create a confusion matrix 
# and display it with ConfusionMatrixDisplay, passing display_labels=iris.target_names. Save the figure to
# outputs/knn_confusion_matrix.png. Add a comment: which pair of species does the model most often confuse (if any)?
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.title("KNN Confusion Matrix (Iris)")
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/knn_confusion_matrix.png")
plt.show()
# Q:Add a comment: which pair of species does the model most often confuse (if any)?
# A: No there are such pair in this model

# --- Decision Trees ---  Q1
print("\nDecision Trees Question 1:\n")
# Create a DecisionTreeClassifier(max_depth=3, random_state=42),
#  fit it on the unscaled training data, and predict on the test set. 
# Print the accuracy score and classification report.
#  Add a comment comparing the Decision Tree accuracy to KNN.
#  Then add a second comment: given that Decision Trees don't rely
#  on distance calculations, would scaled vs. unscaled data affect the result?
dectree = DecisionTreeClassifier(max_depth=3, random_state=42)

dectree.fit(X_train, y_train)

preds = dectree.predict(X_test)
dec_accuracy_score=accuracy_score(y_test, preds)
print("Accuracy:", accuracy_score(y_test, preds))
print("\nfull classification report:\n",classification_report(y_test, preds))

print(    "Accuracy of knn better than decision tree"
    if knn_accuracy_score > dec_accuracy_score
    else "Accuracy of decision tree better than knn")
# knn perform better then decision tree
# scaling should not affect performance of decision trees because the split done by using
# tresholds on single features.

# --- Logistic Regression ---   Q1 
print("Logistic Regression Question 1\n")


# Q:
# Train three logistic regression models on the scaled Iris data,
# identical in every way except for the C parameter: C=0.01, C=1.0, and C=100.
# Use max_iter=1000 and solver='liblinear' for all three. For each model, 
# print the C value and the total size of all coefficients using np.abs(model.coef_).sum(). 
# Add a comment: what happens to the total coefficient magnitude as C increases?
# What does this tell you about what regularization is doing?

# Iris has 3 classes; liblinear only handles binary LR, so I use OneVsRestClassifier
# (one liblinear model per class), as sklearn recommends for multiclass + liblinear.
for C in (0.01, 1.0, 100.0):
    log_reg_model = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000, random_state=42, C=C, solver="liblinear"
        )
    )
    log_reg_model.fit(X_train_scaled, y_train)
    coef_size = sum(np.abs(est.coef_).sum() for est in log_reg_model.estimators_)
    print(f"C value is {C:6g} size of all coefficients = {coef_size:.6f}")

# As C increases, the total coefficient magnitude increases.
# This shows regularization is shrinking coefficients when C is small.

# PCA
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

# PCA Question 1
print("\nPCA Question 1\n")
print(f"X_digits shape: {X_digits.shape}\nimages shape: {images.shape}")
# X_digits shape: (1797, 64)
# images shape: (1797, 8, 8)

# Then create a 1-row subplot showing one example of each digit class (0-9),
# using cmap='gray_r' with each digit's label as the title.
# Save the figure to outputs/sample_digits.png.
# (gray_r is the reversed grayscale colormap -- it renders higher pixel values as darker,
# so digits appear as dark ink on a light background, which is more readable than the default.)
fig, axes = plt.subplots(1, 10, figsize=(10, 2))
for i in range(10):
    index = np.where(y_digits == i)[0][0]
    axes[i].imshow(images[index], cmap="gray_r")
    axes[i].set_title(str(i))
    axes[i].axis("off")
plt.suptitle("One sample per digit (0-9), Digits dataset", y=1.1)
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/sample_digits.png", bbox_inches="tight")
plt.show()

# PCA Question 2
print("PCA Question 2\n")
# Fit PCA() on X_digits (with no n_components argument)
pca = PCA()
pca.fit(X_digits)
# then get the scores with scores = pca.transform(X_digits).
scores = pca.transform(X_digits)
# As in the lesson, scores tell you how strongly each component
# is weighted for each sample -- scores[i, 0] is the weighting
# for PC1 in sample i, scores[i, 1] is the weighting for PC2, and so on.

# Use scores[:, 0] and scores[:, 1] to make a scatter plot,
# coloring each point by its digit label and adding a colorbar.
# Here is the pattern for coloring by a label array and attaching a colorbar:

scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap="tab10", s=10)
plt.colorbar(scatter, label="Digit")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Digits: first two principal components")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/pca_2d_projection.png", bbox_inches="tight")
plt.show()

# Q: do same-digit images tend to cluster together in this 2D space?
# A: Same-digit images often form clusters in 2D 
# but some classes overlap because two PCs do not capture all separation.

# PCA Question 3
print("PCA Question 3\n")
# Using the PCA object you fit in Question 2, plot cumulative explained variance
# vs. number of components using np.cumsum(pca.explained_variance_ratio_).
# Save to outputs/pca_variance_explained.png.
cumul_var = np.cumsum(pca.explained_variance_ratio_)
n = np.arange(1, len(cumul_var) + 1)
n_for_80 = int(np.searchsorted(cumul_var, 0.8, side="left")) + 1
plt.figure(figsize=(8, 4))
plt.plot(n, cumul_var, marker="o", ms=2)
plt.axhline(0.8, color="gray", linestyle="--", label="80%")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance ratio")
plt.title("PCA: cumulative explained variance (Digits)")
plt.xlim(0, len(n))
plt.ylim(0, 1.02)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/pca_variance_explained.png", bbox_inches="tight")
plt.show()

# Q: approximately how many components do you need to explain 80% of the variance?
# A: Aproximately 13 components needed for 80% on Digits; most variance is in the first PCs.
print(f"Components to reach >=80% explained variance: {n_for_80}\n")

# PCA Question 4
print("PCA Question 4\n")
# The preprocessing lesson showed that a reconstruction is built by starting
# from the mean and adding each component weighted by its score. 
# Here is the same idea generalized to n components -- add this function to your file:


def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)


# Using this function, the PCA object, and the scores from Question 2,
# reconstruct the first 5 digits in X_digits using reconstruction through principal components n = 2, 5, 15, and 40.
# Build a grid of subplots where rows correspond to each n value and columns show those 5 digits. 
# Add an "Original" row at the top (use images[i], which is already shaped as (8, 8)). 
# Save to outputs/pca_reconstructions.png.

n_recon_list = (2, 5, 15, 40)
n_cols = 5
n_rows = 1 + len(n_recon_list)
row_labels = ("Original",) + tuple(f"n = {k}" for k in n_recon_list)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.8))
for r, row_label in enumerate(row_labels):
    for c in range(n_cols):
        ax = axes[r, c]
        if r == 0:
            ax.imshow(images[c], cmap="gray_r")
            ax.set_title(f"sample {c}", size=8)
        else:
            n_comp = n_recon_list[r - 1]
            ax.imshow(reconstruct_digit(c, scores, pca, n_comp), cmap="gray_r")
        ax.set_xticks([])
        ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(row_label, rotation=90, size=9, labelpad=8)
plt.suptitle("PCA reconstructions (first 5 images in dataset)", y=1.01)
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/pca_reconstructions.png", bbox_inches="tight")
plt.show()

#Q: at what n do the digits become clearly recognizable, 
#A: very close to the input by n=40.
# and does that match where the variance curve levels off?
# This matches the variance curve: the steep rise is in the first 10–15 components 
