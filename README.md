# Iris Dataset Visualizations

This project provides a comprehensive visual analysis of the **Iris dataset** using various types of plots to understand relationships between the different features (`sepal_length`, `sepal_width`, `petal_length`, and `petal_width`) and how they vary across species (`setosa`, `versicolor`, `virginica`).

---

## 📊 Visualizations Included

### 1. Density Plot of Petal Length by Species
- Shows the distribution of petal length values for each species.
- Highlights clear separation between species based on petal length.

### 2. Bubble Chart: Sepal Dimensions with Petal Length as Size
- X-axis: Sepal Length  
- Y-axis: Sepal Width  
- Bubble Size: Petal Length  
- Color: Species  
- Useful for multivariate analysis combining shape, size, and species.

### 3. Box Plot of All Features
- Displays the spread and outliers of all numerical features.
- Useful to detect range and central tendency.

### 4. Box Plot of Petal Length by Species
- Shows variation in petal length for each species with median, quartiles, and outliers.

### 5. Pair Plot of Iris Dataset
- Pairwise scatter plots for all feature combinations.
- Diagonal contains KDE distributions.
- Clearly shows clusters and linear separability between species.

### 6. Histograms of Features
- Separate histograms for each feature: sepal length, sepal width, petal length, petal width.
- Displays the frequency distribution of values.

### 7. Scatter Plot: Sepal Length vs Sepal Width
- Species are colored differently.
- Illustrates the relationship and spread of sepal dimensions across classes.

---

## 📁 Dataset
The dataset used is the **Iris dataset**, one of the most well-known datasets in pattern recognition. It contains 150 samples with the following features:
- `sepal_length` (cm)
- `sepal_width` (cm)
- `petal_length` (cm)
- `petal_width` (cm)
- `species` (setosa, versicolor, virginica)

---

## 🔧 Tools Used
- Python
- Pandas
- Matplotlib
- Seaborn

---

## 📌 How to Use
1. Clone the repo or open the notebook.
2. Ensure dependencies are installed:  
   ```bash
   pip install pandas matplotlib seaborn
3. Run the notebook to generate visualizations.
4. Use the plots to explore relationships, distribution, and classification potential of the dataset.

# 📷 Example Output

Below are some sample visualizations generated from the notebook:

<img width="1920" height="1048" alt="Screenshot from 2025-08-06 18-47-41" src="https://github.com/user-attachments/assets/17f01d2d-20f1-4b13-b0bc-d6c9a9558bf8" />
<img width="1920" height="1048" alt="Screenshot from 2025-08-06 18-47-51" src="https://github.com/user-attachments/assets/e9b4a231-9908-44b3-ad6e-69f46e59afb8" />
<img width="1920" height="1080" alt="Screenshot from 2025-08-06 19-04-08" src="https://github.com/user-attachments/assets/933c933b-f855-4f2d-928c-7b389bfe5948" />
<img width="1920" height="1080" alt="Screenshot from 2025-08-06 19-03-38" src="https://github.com/user-attachments/assets/6ef9f45f-0245-42c2-a8dd-7d49cb13b9a5" />
<img width="1920" height="1080" alt="Screenshot from 2025-08-06 19-03-54" src="https://github.com/user-attachments/assets/fd9dacb3-0394-4e07-8dfe-82db77f58bf4" />
<img width="1920" height="1080" alt="Screenshot from 2025-08-06 19-04-20" src="https://github.com/user-attachments/assets/de4fe700-dda8-4a23-af4b-e97f431aef35" />
