# Step 1: Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix

# Step 2: Load the Iris dataset
iris = sns.load_dataset("iris")
print(iris.head())

# Step 3: Histogram
iris.hist(edgecolor='black', figsize=(10, 6))
plt.suptitle("Histogram of Iris Dataset Features")
plt.show()

# Step 4: Scatter Plot (sepal_length vs sepal_width)
sns.scatterplot(data=iris, x="sepal_length", y="sepal_width", hue="species")
plt.title("Scatter Plot: Sepal Length vs Sepal Width")
plt.show()

# Step 5: Scatter Matrix (Pair Plot)
sns.pairplot(iris, hue="species", height=2.5)
plt.suptitle("Pair Plot of Iris Dataset", y=1.02)
plt.show()

# Step 6: Box Plot (All features)
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris)
plt.title("Box Plot of All Features")
plt.show()

# Box Plot: Petal Length by Species
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris, x="species", y="petal_length")
plt.title("Box Plot of Petal Length by Species")
plt.show()

# Step 7: Density Plot (KDE)
plt.figure(figsize=(10, 6))
for species in iris["species"].unique():
    sns.kdeplot(data=iris[iris["species"] == species]["petal_length"], label=species, fill=True)
plt.title("Density Plot of Petal Length by Species")
plt.xlabel("Petal Length")
plt.legend()
plt.show()

#  Step 8: Bubble Chart
plt.figure(figsize=(10, 6))
bubble_size = iris["petal_length"] * 30  # Scaling bubble size for visibility
sns.scatterplot(
    data=iris,
    x="sepal_length",
    y="sepal_width",
    size=bubble_size,
    hue="species",
    alpha=0.6,
    sizes=(20, 400),
    legend=False
)
plt.title("Bubble Chart: Sepal Dimensions with Petal Length as Size")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

