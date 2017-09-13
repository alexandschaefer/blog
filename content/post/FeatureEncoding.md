+++
date = "2017-04-13T10:00:00"
draft = false
tags = []
title = "Basic Feature Encoding"
math = true
+++

Recently, I wanted to make a blog post on prediction problems I encounter in my work as data scientist at [project-a](https://www.project-a.com/). While working on the post I realized that feature encoding  might be too basic for some readers; while very much needed for others. As a consequence I decided feature encoding should become its own post.

What is **feature encoding**? Not all information we have at hand is suited for the machine learning algorithm we want to use. Therefore, we have to employ a mapping or encoding e.g. from text to numbers. Here, I want to explain some basice feature encoding and give examples
in python.

### Two Types of Features

Usually you encounter two types of features: numerical or categorical.

 1.  Numerical features are usually real or integer numbers. Example numerical features are
*revenue of a customer*, *days since last order* or *number of orders*.
 2.  Categorical features are often given as text or boolean variables. Examples for categorical
 features are *first product category*, *first marketing channel* or *order delivery type*.

While numerical features are often straight forward to integrate, categorical features need a bit more work. Most machine learning algorithms do not understand text so we need to encode the text into numbers.

| Categorical Feature | Encoded Feature               |
| -------------------| ------------------------------ |
| `"Text A"`             | 0            |
| `"Text B"`    | 1             |
| `"Text C"`    | 2             |

Training the encoder with [sklearn](http://scikit-learn.org) can be as simple as this:

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(list(data.values.flatten()) )
```
And then applying the [label encoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) onto the data
```python
le.transform(list(data.values))
```

This ensures a one to one mapping and makes the categorical features understandable for our ML algorithm. However,
a numbering  like $$0,1,2, \dotsc $$ implies an ordering e.g. $$ 0 < 1 < 2 < \dotsc .$$ Which also implies that
$$ " \mathrm{Text \enspace A}" < "\mathrm{Text \enspace B}" < "\mathrm{Text \enspace C}" < \dotsc ,$$ which we do not know and therefore not want to imply.

There is a solution to this problem which is a binary endcoding also called [one hot encoding](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).

| Categorical Feature | Encoded Feature               | One Hot Encoded Feature
| -------------------| ------------------------------ |------------------------------ |
| `"Text A"`    | 0             |(1,0,0,...)
| `"Text B"`    | 1             |(0,1,0,...)
| `"Text C"`    | 2             |(0,0,1,...)

In python this can look like this

```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse=False,n_values=np.max(data)+1)
enc.fit(data)
```

So let us assume you have $n$ different categories you will create $n$-dimensional binary feature vectors. In each dimension the feature do not create an ordering and only
contain the information of beeing present or not. This also came at a price as you created $n-1$ new feature dimensions. 


### Production Setting

In a production setting you have to keep training and testing as separated processes. Often training is very expensive (in terms of computing time) so you want to do it as few times
as possible. However, you have to use the **same encoding in training and testing** process. As as a result you may want to save the encoder you used in the training setting, and load this encoder in the testing setting using [joblib](http://scikit-learn.org/stable/modules/model_persistence.html)  
```python
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

if train : ## fit and write encoder
   le = LabelEncoder()
   le.fit(list(data.values.flatten()) ) # we want one encoder for everything
   joblib.dump(le, '../model/label_encoder.pkl')
else : ## read encoder
   le = joblib.load('../model/label_encoder.pkl') 
```
This scheme enforces a strict separation of train and test data. Which is a quality in itself I do often prefer. However, if your test data encloses a feature which was not present in your training this approach will fail. This cannot be fixed by fitting a new encoder without re-running the training step. Soutions might range from ignoring unknown features to constructing complex [ETL](https://en.wikipedia.org/wiki/Extract,_transform,_load) processes.
 