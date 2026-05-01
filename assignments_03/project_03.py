import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import requests
from io import BytesIO

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=RuntimeWarning)
COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]
# Task 1: Load and Explore
# Adapt that code for your script.
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES
print(df.head())


# Once it is loaded, take some time to understand what you are working with.
# How many emails are in the dataset?
print("Total emails:", len(df))
#  How balanced are the two classes?
counts = df["spam_label"].value_counts()
print("Class counts:\n", counts)
print("Class %:\n", (counts / len(df) * 100).round(2))
# What does that balance (or imbalance) mean for how you should interpret a raw accuracy score?
# baseline accuracy if you always predict the majority class (ham)
print("Majority-class baseline accuracy:", (counts.max() / len(df)).round(3))
# Now explore how a few key features differ between spam and ham.
# For each of word_freq_free, char_freq_!, and capital_run_length_total,
# create a boxplot showing the distribution of that feature for spam emails versus ham emails.
#  Save them to outputs/.
# Q:  What do you notice? Are the differences between classes dramatic or subtle?
# A: The differences more on the subtle side not dramaticly different.
#    Neither of the fatures can't tell alone abou spam or ham for sure.
#  
os.makedirs("outputs", exist_ok=True)
features_to_plot = ["word_freq_free", "char_freq_!", "capital_run_length_total"]
for feature in features_to_plot:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="spam_label", y=feature)
    plt.xlabel("Class (0 = ham, 1 = spam)")
    plt.title(f"{feature}: spam vs ham")
    plt.tight_layout()
    safe_name = feature.replace("!", "bang").replace("$", "dollar").replace(";", "semicolon")
    plt.savefig(f"outputs/boxplot_{safe_name}.png", bbox_inches="tight")
    plt.show()

# Q: Then look at the raw scale of the features more broadly. Notice that many emails have a value
#  of zero for most word-frequency features -- most emails do not contain the word "free" at all. 
# What does this heavy skew toward zero tell you about the data? Why does the numeric scale vary so 
# dramatically across features (some are tiny fractions,
#  others reach into the thousands)?
# A: The data that we measuring and comparing is different in terms that 
# it different scales different things that we measure like quantity of capital letters (thousands)
#  and sum of free word in whole letter

#  Q: Why might that matter for some of the models you are about to build
# A: It matters to the models because the big differences between values of the features can dominate one of  them

print("Task 2: Prepare Your Data")

# Before building any models, prepare your data for the experiments in Task 3. You will need a train/test split
# and will need to think about how to handle the feature scales you noticed in Task 1.
#  Q: Document your choices in comments.
#  A: the feature scaling done by standardizing them so each column will have the mean 0 and std 1
#     and use only x_train to fit to avoid data leakage.

X = df.drop("spam_label", axis=1)
y = df["spam_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA preprocessing
# Not every classifier benefits from dimensionality reduction.
#  Decision trees and random forests split on feature thresholds -- they are insensitive to feature scale or correlation,
#  so PCA is unlikely to help them.
#  KNN and logistic regression are different: both operate in a space where feature magnitudes matter
#  and can benefit from reduced dimensionality.

# One rule applies whenever you use PCA: always scale the data first.
#  PCA finds directions of maximum variance, so features with larger raw values will dominate
#  unless you standardize first -- the same reason scaling is often used for KNN.
#  For Spambase, where word frequencies are tiny fractions and capital_run_length_total can reach the thousands, 
# this ordering is essential.

# Fit PCA on the training data only -- same reason as the scaler: fitting on all the data lets test-set 
# information leak into the components.

pca = PCA()
pca.fit(X_train_scaled)
# Plot the cumulative explained variance, save it to outputs/, and print n -- the number of components where it first reaches 90%.
cumul_var = np.cumsum(pca.explained_variance_ratio_)
n = int(np.searchsorted(cumul_var, 0.9, side="left")) + 1

plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(cumul_var) + 1), cumul_var, marker="o", ms=2)
plt.axhline(0.9, color="gray", linestyle="--", label="90%")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance ratio")
plt.title("PCA: cumulative explained variance (Spambase, scaled)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/pca_variance_explained_spambase.png", bbox_inches="tight")
plt.show()

print(f"Components to reach >=90% explained variance: {n}")

# With n determined, transform both sets and slice to the first n components:

X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca  = pca.transform(X_test_scaled)[:, :n]
# Keep both the full scaled arrays and the PCA-reduced arrays -- you will use both in Task 3.

print("Task 3: A Classifier Comparison")
# Build and evaluate the following five classifiers.
#  For each, print the accuracy and the full classification report.

# KNeighborsClassifier(n_neighbors=5) trained on the unscaled data
knn_on_unscaledData = KNeighborsClassifier(n_neighbors=5)
knn_on_unscaledData.fit(X_train, y_train)
preds = knn_on_unscaledData.predict(X_test)
knn_acc_unscaled =accuracy_score(y_test, preds)
print("Accuracy knn_on_unscaledData:", knn_acc_unscaled)
print("\nKNN Classifier full classification report:\n",classification_report(y_test, preds))

# KNeighborsClassifier(n_neighbors=5) trained on the scaled data, 
knn_on_scaledData = KNeighborsClassifier(n_neighbors=5)
knn_on_scaledData.fit(X_train_scaled, y_train)
preds = knn_on_scaledData.predict(X_test_scaled)
knn_acc_scaled=accuracy_score(y_test, preds)
print("Accuracy knn_on_scaledData:", knn_acc_scaled)
print("\nKNN Classifier full classification report:\n",classification_report(y_test, preds))

# PCA-reduced data from Task 2 -- compare the two
knn_on_PCAData = KNeighborsClassifier(n_neighbors=5)
knn_on_PCAData.fit(X_train_pca, y_train)
preds = knn_on_PCAData.predict(X_test_pca)
knn_acc_pca=accuracy_score(y_test, preds)
print("Accuracy knn_on_PCAData:", knn_acc_pca)
print("\nKNN Classifier full classification report:\n",classification_report(y_test, preds))

if knn_acc_pca < knn_acc_scaled - 1e-3:
    pca_vs_scaled_msg = (
        "Scaled KNN is slightly better than PCA KNN, so PCA likely removed some useful information "
    )
elif knn_acc_pca > knn_acc_scaled + 1e-3:
    pca_vs_scaled_msg = "PCA KNN is slightly better than scaled KNN here."
else:
    pca_vs_scaled_msg = (
        "Scaled KNN and PCA KNN are very close, so PCA is a reasonable speed/compression tradeoff with little accuracy loss."
    )

print(
    f"Overall test accuracy: unscaled {knn_acc_unscaled:.4f} vs scaled {knn_acc_scaled:.4f} vs PCA {knn_acc_pca:.4f}. "
    f"{pca_vs_scaled_msg}"
)
# DecisionTreeClassifier(random_state=42) -- before settling on a final depth,
# try max_depth values of 3, 5, 10, and None (unlimited).
# For each, print both the training accuracy and the test accuracy.
depths = [3, 5, 10, None]
results = []
for depth in depths:
    dectree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dectree.fit(X_train, y_train)

    train_preds = dectree.predict(X_train)
    test_preds = dectree.predict(X_test)
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    results.append((depth, train_acc, test_acc))

    depth_label = "None" if depth is None else str(depth)
    print(f"DecisionTree max_depth={depth_label:>4}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

# Q: What do you notice as depth increases? What does that tell you about overfitting?
# A: As depth increases the accuracy increases a lot too. The model becoming more complex and can overfit. 
# Q: Pick the depth you would use in production and add a comment explaining your reasoning.
# A: The depth with the best accuracy is None so I would choose it
# Then, using your chosen depth, print the accuracy and full classification report as you did for the other classifiers.

best_depth, _, _ = max(results, key=lambda t: (t[2], -999 if t[0] is None else -t[0]))
best_depth_label = "None" if best_depth is None else str(best_depth)
print(f"\nChosen depth for production (by best test accuracy): {best_depth_label}")

dectree_best = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dectree_best.fit(X_train, y_train)
preds = dectree_best.predict(X_test)
print("Accuracy DecisionTree (chosen):", accuracy_score(y_test, preds))
print("\nfull classification report for decision trees:\n", classification_report(y_test, preds))

# LogisticRegression(C=1.0, max_iter=1000, solver='liblinear') trained on the scaled data,
# and again on the PCA-reduced data 

# LogisticRegression on scaled data
log_reg_scaled = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=42)
log_reg_scaled.fit(X_train_scaled, y_train)
preds_lr_scaled = log_reg_scaled.predict(X_test_scaled)

print("Accuracy LogisticRegression scaled:", accuracy_score(y_test, preds_lr_scaled))
print("\nLogisticRegression scaled report:\n", classification_report(y_test, preds_lr_scaled))

# LogisticRegression on PCA-reduced data
log_reg_pca = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=42)
log_reg_pca.fit(X_train_pca, y_train)
preds_lr_pca = log_reg_pca.predict(X_test_pca)

print("Accuracy LogisticRegression PCA:", accuracy_score(y_test, preds_lr_pca))
print("\nLogisticRegression PCA report:\n", classification_report(y_test, preds_lr_pca))


acc_scaled = accuracy_score(y_test, preds_lr_scaled)
acc_pca = accuracy_score(y_test, preds_lr_pca)
print("\nComparison:", "Scaled >= PCA" if acc_scaled >= acc_pca else "PCA > Scaled")
# Q: -- compare the two
# A: Logistic Regression scaled performed better with accuracy of 0.929 then Logistic Regression PCA with accuracy of 0.918

# RandomForestClassifier (introduced below)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Both the Decision Tree and the Random Forest expose a .feature_importances_ attribute.
# After building both, print the top 10 most important features for each
# and save a bar chart of the Random Forest importances to outputs/feature_importances.png.


preds_rf = rf.predict(X_test)
print("Accuracy RandomForest:", accuracy_score(y_test, preds_rf))
print("\nRandomForest report:\n", classification_report(y_test, preds_rf))

feature_names = X.columns.to_list()
dt_importances = pd.Series(dectree_best.feature_importances_, index=feature_names).sort_values(ascending=False)
rf_importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

print("\nTop 10 DecisionTree feature importances:")
print(dt_importances.head(10))

print("\nTop 10 RandomForest feature importances:")
print(rf_importances.head(10))



plt.figure(figsize=(10, 5))
rf_importances.head(10).sort_values().plot(kind="barh")
plt.title("Random Forest: Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/feature_importances.png", bbox_inches="tight")
plt.show()

# Do the two models agree on which features matter most? Do the results match your intuition
# about what makes an email spam?
# A: both models agree on the features that make the email spam:
#  high "$" sign and "remove" and "!" but the order is different


# In Task 4 you will cross-validate all your models --
#  the variance across folds for the Random Forest should be noticeably lower than for the Decision Tree.

print("Task 4: Cross-Validation")
# Using cross_val_score with cv=5 (same style as warmup_03.py), evaluate models on the training split.
# For each, print the mean and standard deviation of the fold scores.


cv = 5

cv_jobs = [
    ("KNN (unscaled)", KNeighborsClassifier(n_neighbors=5), X_train, y_train),
    ("KNN (scaled)", KNeighborsClassifier(n_neighbors=5), X_train_scaled, y_train),
    ("KNN (PCA)", KNeighborsClassifier(n_neighbors=5), X_train_pca, y_train),
    (
        f"DecisionTree (depth={best_depth_label})",
        DecisionTreeClassifier(max_depth=best_depth, random_state=42),
        X_train,
        y_train,
    ),
    ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42), X_train, y_train),
    (
        "LogReg (scaled)",
        LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=42),
        X_train_scaled,
        y_train,
    ),
    (
        "LogReg (PCA)",
        LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=42),
        X_train_pca,
        y_train,
    ),
]

cv_summary = []
print(f"\n5-fold CV on training split (n={len(X_train)})")
for name, model, X_cv, y_cv in cv_jobs:
    scores = cross_val_score(model, X_cv, y_cv, cv=cv, scoring="accuracy")
    mean_score = float(scores.mean())
    std_score = float(scores.std())
    cv_summary.append((name, mean_score, std_score))
    print(f"{name:25s}  mean={mean_score:.4f}  std={std_score:.4f}  folds={np.round(scores, 4)}")

most_accurate = max(cv_summary, key=lambda t: t[1])
most_stable = min(cv_summary, key=lambda t: t[2])
print(f"\nMost accurate (mean CV): {most_accurate[0]}  mean={most_accurate[1]:.4f}  std={most_accurate[2]:.4f}")
print(f"Most stable (lowest std): {most_stable[0]}  mean={most_stable[1]:.4f}  std={most_stable[2]:.4f}")

print("\nWrite-up helpers (based on this run):")
print(
    f"- Most accurate: {most_accurate[0]} (mean={most_accurate[1]:.4f}, std={most_accurate[2]:.4f})\n"
    f"- Most stable:   {most_stable[0]} (mean={most_stable[1]:.4f}, std={most_stable[2]:.4f})"
)

print("Task 5: Building a Prediction Pipeline")

# knn5_pipeline = Pipeline([
#     ("scaler",     StandardScaler()), # name, object pattern
#     ("classifier", KNeighborsClassifier(n_neighbors=5))
# ])
# knn5_pipeline.fit(X_train, y_train)
# y_pred = knn5_pipeline.predict(X_test)

# pca_pipeline = Pipeline([
#     ("scaler",     StandardScaler()),
#     ("pca",        PCA(n_components=n)),  # n chosen in Task 2 (>=90% variance)
#     ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")),
# ])
# pca_pipeline.fit(X_train, y_train)
# y_pred_pca_pipe = pca_pipeline.predict(X_test)
# print("Accuracy PCA+LogReg pipeline:", accuracy_score(y_test, y_pred_pca_pipe))

# Build your pipelines
# Build two pipelines: one for your best tree-based classifier
# and one for your best non-tree-based classifier. 
# For each, fit on the training data and print the full classification report on the test set.
# Confirm the results match your earlier manual approach. 
# If your Task 3 experiments showed that PCA improved your non-tree model, include it as a step in that pipeline.

# Best tree-based is RandomForest 
tree_pipeline = Pipeline(
    [("classifier", RandomForestClassifier(n_estimators=100, random_state=42))]
)
tree_pipeline.fit(X_train, y_train)
y_pred_tree_pipe = tree_pipeline.predict(X_test)
print("\nTask 5 - RandomForest pipeline test classification report:")
print(classification_report(y_test, y_pred_tree_pipe))
print(
    "Match manual RF test accuracy?",
    np.isclose(accuracy_score(y_test, y_pred_tree_pipe), accuracy_score(y_test, preds_rf)),
)

# Best non-tree is scaled LogisticRegression
non_tree_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=42),
        ),
    ]
)
non_tree_pipeline.fit(X_train, y_train)
y_pred_non_tree_pipe = non_tree_pipeline.predict(X_test)
print("\nTask 5 - Scaled LogisticRegression pipeline test classification report:")
print(classification_report(y_test, y_pred_non_tree_pipe))
print(
    "Match manual LogReg(scaled) test accuracy?",
    np.isclose(
        accuracy_score(y_test, y_pred_non_tree_pipe),
        accuracy_score(y_test, preds_lr_scaled),
    ),
)

# Q: do they have the same structure? 
# A: They do not have the same structure: the non-tree pipeline includes scaling 
# and could include PCA if it will improve that model, while the tree pipeline is just the forest.
# Q: Why or why not? What is the practical value of packaging a model this way,#  especially when handing it off to someone else or deploying it?
# A: A Pipeline bundles preprocessing + model so fit/predict always apply the same steps,
#    which reduces mistakes when saving, sharing, or deploying.