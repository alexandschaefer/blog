+++
date = "2017-04-13T10:00:00"
draft = false
tags = []
title = "Predicting Customer Behavior - A General Approach"
math = true
+++

## Predicting Customer Behavior - A General Approach

In my work as a data scientist at [Project-A](https://www.project-a.com/) I often encounter the task of predicting customer behavior. For many businesses it can be very valuable to estimate which one of their new customers will be the most profitable in the future. Or which one of the existing customers will stop being valuable. In this context many tasks such as predicting churn, customer lifetime value or ending of newsletter subscription are of interest. While these tasks sound very different the principles can be easily generalized to any [predictive modeling](https://en.wikipedia.org/wiki/Predictive_analytics) task. Here, I will evaluate on the general principle.

### Time scale

Usually, it is not of interest to predict any arbitrary timepoint in the future. It is sufficient to estimate the target variable of interest for a specific point in time, e.g. the behavior of a customer in 12 months from now. This makes the modeling of our problem a lot easier. Instead of forecasting a complex timeseries we only need to make a single **point estimate**. 

To formalize this a little bit. We want to predict the customer behavior in $t$ months from now. And denote the customer behavior as $y_t$ and our prediction with $\hat{y_t}$. We do this prediction based on our current observations of the customer $x_0$ (e.g. his buying frequency, his created revenue up to date) . 

### Data

Data is the key component in predictive modeling. We want to make a prediction of the future based on our observations in the past. So in order to train a predictive model that predicts how customers behave in $t$ months from now. We actually go $t$ months in to the past look at our observation $x_{-t}$ and build a model $m_{now}$ that predicts our current customer behavior $y_{now}$
$$
\underset{m_{now}}{\mathrm{argmin}} (\hat{y}_{now} - y_{now}) \\
\underset{m_{now}}{\mathrm{argmin}} (f(x_{-t},m_{now}) - y_{now}).
$$
We do also add a [regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics)) term $R(f)$ as our aim is not learning the training data perfectly but to [generalize](https://en.wikipedia.org/wiki/Generalization_error) as good as possible
$$
\underset{m_{now}}{\mathrm{argmin}} \bigg(\Big(f(x_{-t},m_{now}) - y_{now}\Big) - \lambda \Big(R(f)\Big)\bigg).
$$

###  Validation

The classical way of validating machine learning models is cross validation. For cross validation we estimate or train our model on a subset of the data. And estimate its performance on the remaining set. We can do this multiple time so that each customer was at some point part of the training dataset and at some point part of the testing dataset. This is good because it tells us how well we generalize towards new customers. 

However, here we are not so much interested in how well our predictive model generalizes to new customers. Rather we are interested how well our model generalizes over time.  Therefore we might want to validate our approach by learning a model $m_{-t}$ that trains on observation data $x_{-2t}$:
$$
\underset{m_{-t}}{\mathrm{argmin}} \bigg(\Big(f(x_{-2t},m_{-t}) - y_{-t}\Big) - \lambda \Big(R(f)\Big)\bigg).
$$
This model is ideal to validate the temporal generalization of our approach. We can apply it onto our observation at timepoint $-t$ to predict our current observation:
$$
f(x_{-t},m_{-t}) = \hat{y}_{now} .
$$
The difference between our prediction and the actual customer behavior:
$$
{y}_{now} - \hat{y}_{now} ,
$$
can be a good estimate how our approach generalizes over time. 



This model is ideal to validate

Most imported points:

Validation (Historic or A/B)

Find out what Stakeholder really wants

How will the result be used 

