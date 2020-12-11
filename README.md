# Introduction:

My research question is to see how accurately a machine learning model can classify the type of a star based on attributes of the star such as temperature and color. This is interesting because there are more stars visible to our eyes than we can ever hope to count. A model that can accurately classify stars would greatly help our efficiency as we try to document the cosmos.


```python
import pandas as pd 
import numpy as np 
import requests
import matplotlib as plt 
import seaborn as sns 
import pandas_profiling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
```

# Data:

- My data is found at the URL below. The data is a table of approximately 1200 rows, with different features of stars including temperature, mass, radius, and color.  
- To collect the data, I am reading the html and pulling out each row of the table, and appending it to a dataframe.  
- I then clean the data, specifically turning the RGB color into a decimal.
- My code also generates an html file containing the exploratory data analysis.


```python
url = "http://www.isthe.com/chongo/tech/astro/HR-temp-mass-table-byhrclass.html#below"
```


```python
# Read the star data in from the web
star_data = pd.DataFrame()
html = pd.read_html(url)
for i in html:
    star_data = pd.concat([star_data, i])
```


```python
len(star_data)
```




    1242




```python
# Drop unneededd columns
star_data.drop([0, 1, 2], axis=1, inplace=True)
```


```python
# Remove all empty values
star_data.dropna(axis=0, inplace=True)
```


```python
len(star_data)
```




    1100




```python
star_data['color_decimal'] = ""
```


```python
# Function that takes RGB values as a string, turning them into a decimal representation of the color
def rgb_cleaner(rgb_text):
    temp_split = rgb_text.split()
    red = int(temp_split[0])
    green = int(temp_split[1])
    blue = int(temp_split[2])
    color_hex = '%02x%02x%02x' % (red, green, blue)
    color_finished = int(color_hex, 16)
    return color_finished

```


```python
# Calling the RGB conversion function on each color listed in the data
star_data['color_decimal'] = [rgb_cleaner(i) for i in star_data['Star ColorRGB 0-255']]
```


```python
# Get rid of the old color data
star_data.drop(['Star ColorRGB 0-255'], axis=1, inplace=True)
```


```python
#The first letter of the star type is the star family. We will use this as our target
star_data['star_family'] = [i[0] for i in star_data['StellarType']]
```


```python
star_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Abs MagMv</th>
      <th>Bolo CorrBC(Temp)</th>
      <th>Bolo MagMbol</th>
      <th>Color IndexB-V</th>
      <th>LuminosityLstar/Lsun</th>
      <th>MassMstar/Msun</th>
      <th>RadiusRstar/Rsun</th>
      <th>StellarType</th>
      <th>TempK</th>
      <th>color_decimal</th>
      <th>star_family</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-9.5</td>
      <td>-4.58</td>
      <td>-14.08</td>
      <td>-0.35</td>
      <td>34100000.0</td>
      <td>160.0</td>
      <td>80.2</td>
      <td>O0Ia0</td>
      <td>50000.0</td>
      <td>9479935</td>
      <td>O</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-6.7</td>
      <td>-4.58</td>
      <td>-11.28</td>
      <td>-0.35</td>
      <td>2590000.0</td>
      <td>150.0</td>
      <td>22.1</td>
      <td>O0Ia</td>
      <td>50000.0</td>
      <td>9479935</td>
      <td>O</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-6.5</td>
      <td>-4.58</td>
      <td>-11.08</td>
      <td>-0.35</td>
      <td>2150000.0</td>
      <td>140.0</td>
      <td>20.2</td>
      <td>O0Ib</td>
      <td>50000.0</td>
      <td>9479935</td>
      <td>O</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-6.5</td>
      <td>-4.58</td>
      <td>-11.08</td>
      <td>-0.35</td>
      <td>2150000.0</td>
      <td>130.0</td>
      <td>20.2</td>
      <td>O0II</td>
      <td>50000.0</td>
      <td>9479935</td>
      <td>O</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-6.5</td>
      <td>-4.58</td>
      <td>-11.08</td>
      <td>-0.35</td>
      <td>2150000.0</td>
      <td>120.0</td>
      <td>20.2</td>
      <td>O0III</td>
      <td>50000.0</td>
      <td>9479935</td>
      <td>O</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Double checking the datatypes of each column before we start running our models
star_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1100 entries, 1 to 1235
    Data columns (total 11 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   Abs MagMv             1100 non-null   float64
     1   Bolo CorrBC(Temp)     1100 non-null   float64
     2   Bolo MagMbol          1100 non-null   float64
     3   Color IndexB-V        1100 non-null   float64
     4   LuminosityLstar/Lsun  1100 non-null   float64
     5   MassMstar/Msun        1100 non-null   float64
     6   RadiusRstar/Rsun      1100 non-null   float64
     7   StellarType           1100 non-null   object 
     8   TempK                 1100 non-null   float64
     9   color_decimal         1100 non-null   int64  
     10  star_family           1100 non-null   object 
    dtypes: float64(8), int64(1), object(2)
    memory usage: 103.1+ KB



```python
import pandas_profiling
```


```python
# This python library generates an html document showing the important features of the data
profile = star_data.profile_report(title='EDA of star dataset using Pandas Profiling')
# save the report
profile.to_file(output_file="star.html")
```

    Summarize dataset: 100%|██████████| 26/26 [00:22<00:00,  1.15it/s, Completed]
    Generate report structure: 100%|██████████| 1/1 [00:07<00:00,  7.19s/it]
    Render HTML: 100%|██████████| 1/1 [00:04<00:00,  4.29s/it]
    Export report to file: 100%|██████████| 1/1 [00:00<00:00, 69.18it/s]



```python
# Set up x and y variables
X = star_data[['Abs MagMv', 'Bolo CorrBC(Temp)', 'Bolo MagMbol', 'Color IndexB-V', 'LuminosityLstar/Lsun', 'MassMstar/Msun', 'RadiusRstar/Rsun', 'TempK', 'color_decimal']]
y = star_data['star_family']
```


```python
# Make training and testing set (testing is 20% of the data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=14)
```

Now it's time to run some models on the data and see how they do:

## Logistic Regression


```python
lr = LogisticRegression()
lr.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)




```python
yhat_train = lr.predict(X_train)
y_pred = lr.predict(X_test)
```


```python
print('Accuracy = {:.5f}'.format(accuracy_score(y_test, y_pred)))
print('Precision = {:.5f}'.format(precision_score(y_test, y_pred, average='weighted')))
print('Recall = {:.5f}'.format(recall_score(y_test, y_pred, average='weighted')))
print('F1 score = {:.5f}'.format(f1_score(y_test, y_pred, average='weighted')))
```

    Accuracy = 0.22727
    Precision = 0.12487
    Recall = 0.22727
    F1 score = 0.13313


## K Nearest Neighbors


```python
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                         weights='uniform')




```python
knn_yhat_train = knn.predict(X_train)
knn_y_pred = knn.predict(X_test)
```


```python
print('Accuracy = {:.5f}'.format(accuracy_score(y_test, knn_y_pred)))
print('Precision = {:.5f}'.format(precision_score(y_test, knn_y_pred, average='weighted')))
print('Recall = {:.5f}'.format(recall_score(y_test, knn_y_pred, average='weighted')))
print('F1 score = {:.5f}'.format(f1_score(y_test, knn_y_pred, average='weighted')))
```

    Accuracy = 0.46364
    Precision = 0.42628
    Recall = 0.46364
    F1 score = 0.43861


## Decision Tree


```python
dt = DecisionTreeClassifier(min_samples_leaf=3)
dt.fit(X_train, y_train)
```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=3, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='best')




```python
dt_yhat_train = dt.predict(X_train)
dt_y_pred = dt.predict(X_test)
```


```python
print('Accuracy = {:.5f}'.format(accuracy_score(y_test, dt_y_pred)))
print('Precision = {:.5f}'.format(precision_score(y_test, dt_y_pred, average='weighted')))
print('Recall = {:.5f}'.format(recall_score(y_test, dt_y_pred, average='weighted')))
print('F1 score = {:.5f}'.format(f1_score(y_test, dt_y_pred, average='weighted')))
```

    Accuracy = 0.50909
    Precision = 0.51527
    Recall = 0.50909
    F1 score = 0.51079


## Random Forest


```python
rf = RandomForestClassifier()
rf.fit(X_test, y_test)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)




```python
rf_yhat_train = rf.predict(X_train)
rf_y_pred = rf.predict(X_test)
```


```python
print('Accuracy = {:.5f}'.format(accuracy_score(y_test, rf_y_pred)))
print('Precision = {:.5f}'.format(precision_score(y_test, rf_y_pred, average='weighted')))
print('Recall = {:.5f}'.format(recall_score(y_test, rf_y_pred, average='weighted')))
print('F1 score = {:.5f}'.format(f1_score(y_test, rf_y_pred, average='weighted')))
```

    Accuracy = 0.91818
    Precision = 0.92038
    Recall = 0.91818
    F1 score = 0.91644


# Methods & Results

- I chose to use 4 different models on my data: Logistic regression, k nearest neighbors, decision trees, and random forest.  
- Out of those 4, random forest performed the best.  
- I think the strength of random forest is that it is an ensemble method. This means that there isn't a single process that runs on the data, but rather several layers of processes. Doing so provides a more accurate result.  
- The limitation of these models is that they all expected numeric features. If I were to use dummy variables I could try a larger variety of models and compare them to each other.  
- I ended up using a lot of the default values for each model. I tried changing a few things like the alpha level or the type of average that was being used but I found that the default values were working pretty well, especially for random forest. With more time, my prediction is that each of these models could be tuned to perform much better.  
- To evaluate my models I used 4 different metrics: accuracy, precision, recall, and the F1 score.  

# Research Question Answer:

I conclude that machine learning models can be trained to classify types of stars, thereby automating a very laborious and near-infinite process.
