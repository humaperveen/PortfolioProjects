# Introduction

Like many other fields, machine learning has enormous significance in the agriculture sector. Dry beans, one of the most consumed legumes, are found in a wide range of varieties all over the world. Classification of various types of dry beans has a great impact on the production of good quality of seeds.


# THE DATASET

We will use the *Dry Beans Dataset* which includes images of 13,611 grains of 7 different registered dry beans taken with a high-resolution camera. A total of 16 features, 12 dimensions and 4 shape forms, were obtained from the grains.

* ATTRIBUTE INFORMATION
1.) Area (A): The area of a bean zone and the number of pixels within its boundaries.

2.) Perimeter (P): Bean circumference is defined as the length of its border.

3.) Major axis length (L): The distance between the ends of the longest line that can be drawn from a bean.

4.) Minor axis length (l): The longest line that can be drawn from the bean while standing perpendicular to the main axis.

5.) Aspect ratio (K): Defines the relationship between L and l.

6.) Eccentricity (Ec): Eccentricity of the ellipse having the same moments as the region.

7.) Convex area (C): Number of pixels in the smallest convex polygon that can contain the area of a bean seed.

8.) Equivalent diameter (Ed): The diameter of a circle having the same area as a bean seed area.

9.) Extent (Ex): The ratio of the pixels in the bounding box to the bean area.

10.)Solidity (S): Also known as convexity. The ratio of the pixels in the convex shell to those found in beans.

11.)Roundness (R): Calculated with the following formula: $(4\pi A)/(P^2)$

12.)Compactness (CO): Measures the roundness of an object: Ed/L

13.)ShapeFactor1 (SF1)

14.)ShapeFactor2 (SF2)

15.)ShapeFactor3 (SF3)

16.)ShapeFactor4 (SF4)

17.)Class (Seker, Barbunya, Bombay, Cali, Dermason, Horoz and Sira)

*   PUBLISHED RESEARCH STUDY BACKGROUND

The publication that we will be considering is:
  > Koklu, M. and Ozkan, I.A., 2020. Multiclass classification of dry beans using computer vision and machine learning techniques. Computers and Electronics in Agriculture, 174, https://doi.org/10.1016/j.compag.2020.105507

# Load Data and Exploratory Data Analysis

*   DOWNLOAD THE DATASET

**Dataset (Dry Bean dataset)can be downloaded from** 
https://archive-beta.ics.uci.edu/dataset/602/dry+bean+dataset

* THE DATASET FOLDER

You can now load the data from the zipped *Pistachio_Image_Dataset* folder. The folder has three sets of data (and so, three subfolders):

  1. *Dry_Bean_Dataset.arff* - i.e., the *Features Version*, with 16 features extracted from the image data, represented as a feature vector and class provided in .arff file formats.
  2.   *Dry_Bean_Dataset.txt* - i.e., the breif information about the dataset.
  3.  *Dry_Bean_Dataset.xlsx* - i.e., the *Features Version*, with 16 features extracted from the image data, represented as a feature vector and class provided in .xlsx file formats.

I will load data from *Dry_Bean_Dataset.xlsx* file.

# Data Pre-processing
1. Changing the classes into a numerical form: The labels provided in the data are in their nominal form (categorical form). These need to be changed to numerical form to apply machine learning algorithms. I used the scikit-learn LabelEncoder module to change classes into a numerical form that Encode target labels with a value between 0 and n_classes-1.

2. Split the data into training, validation, and testing set: I split the dataset into training, validation, and testing sets with the ratio of (70:10:20). 70% data is for training the machine learning models, 10% for hyperparameter tuning using the validation set and 20% is for testing the model performance and for evaluation.

3. Normalization of features: Feature scaling is an essential step in modeling machine learning algorithms. I used the scikit-learn StandardScaler module to normalization of features. Standardization is a scaling technique wherein it makes the data scale-free by converting the statistical distribution of the data into the below format:

mean - 0
standard deviation – 1

$$ 𝒛=\frac {𝑥−𝜇}{𝜎}$$ 

The entire dataset features are standardized by subtracting the mean and scaling to unit variance.

# Classification Models
The dry beans classification is a multiclass classification problem. Various classifiers are followed to considerably address the problem. To improve the accuracy rate, hyperparameter tuning is applied for various parameters.

# Results

I found the accuracy and f1-score of different classifiers as shown in table below. The Accuracy and F1-score for the scikit-learn MLP classifier are highest at 93% and 94% respectively. Three-layer MLP with sigmoid function has lowest accuracy (74.25%) and f1-score (73%).

| Models | Accuracy | F1 Score |
| --- | --- | --- |
| MLP Classifier | 0.9294 | 0.9406 |
| Logistic Regression | 0.9280 | 0.9382 |
| K-Nearest Neighbors | 0.9195 | 0.9328 |
| GaussianNB | 0.9016 | 0.9093|
| SVC | 0.9276 | 0.9395 |
| Random Forest CLassifier | 0.9027 | 0.9109 |
| Decision Tree Classifier | 0.8854 | 0.8901 |
| XGB Classifier | 0.9232 | 0.9354 |
| Three layer MLP | 0.7426 | 0.7297 |


| Models | BARBUNYA | BOMBAY | CALI | DERMASON | HOROZ | SEKER | SIRA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MLP Classifier | 93.44% | 100.0% | 94.24% | 92.18% | 95.73% | 94.62% | 88.24% |
| Logistic Regression | 92.45% | 100.0% | 94.48% | 92.76% | 95.27% | 93.68% | 88.14% | 
| KNN | 92.90% | 100.0% | 94.06% | 91.23% | 95.13% | 93.70% | 85.96% |
| GaussianNB | 84.52% | 99.05% | 89.45% | 90.94% | 94.80% | 92.18% | 85.61% |
| SVM | 93.46% | 100.0% | 94.82% | 92.14% | 95.62% | 94.03% | 87.62% |
| Random Forest | 84.36% | 99.52% | 89.39% | 91.32% | 95.31% | 92.00% | 85.74% |
| Decision Tree | 77.01% | 99.52% | 88.92% | 92.21% | 89.56% | 91.40% | 84.43% | 
| XGBoost | 92.72% | 100.0% | 93.94% | 91.61% | 95.34% | 94.20% | 86.97% |
| Three-layer MLP | 91.40% | 100.00% | 94.83% | 92.41% | 95.30% | 92.98% | 86.55% |

Different models f1-score for various dry beans classes

# Conclusion
The main purpose of this assignment was to evaluate the performance of various machine learning algorithms to automatic detection of uniform seed varieties for more crop production in the agriculture field. Due to the increasing demand for good quality and uniform seed variety, the application of machine learning techniques has been guaranteed helpful to improve the quality of seed for crop production, classification of seeds, and marketing as well.

# References
1. Koklu, M. and Ozkan, I.A., 2020. Multiclass classification of dry beans using computer vision and machine learning techniques. Computers and Electronics in Agriculture, 174, https://doi.org/10.1016/j.compag.2020.105507
2. https://en.wikipedia.org/wiki/Random_forest
3. https://www.researchgate.net/figure/The-structure-of-the-XGBoost-
algorithm_fig3_336649344
4. https://www.javatpoint.com/multi-layer-perceptron-in-tensorflow
Koklu, M. and Ozkan, I.A., 2020. Multiclass classification of dry beans using computer vision and machine learning techniques. Computers and Electronics in Agriculture,
174, https://doi.org/10.1016/j.compag.2020.105507
5. https://scikit- learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
6. https://scikit- learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
7. https://scikit- learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
8. https://scikit- learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
9. https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
10. https://scikit-
learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
11. https://scikit-
learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
12. https://scikit-learn.org/stable/modules/model_evaluation.html
13. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
14. https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-
glr-auto-examples-model-selection-plot-roc-py
15. Build the neural network (Pytorch Tutorial)
https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
