# Predicting Heart Disease Using Machine Learning

**The goal of this project was to develop a machine learning model that could predict heart disease in a random data set of patients, with a 95% accuracy or higher.** 

## The Data Source

The training and testing dataset contains a variety of features that could be considered indicators of a patient’s heart health (age, sex, chest-pain level, etc) and whether or not the patient has heart disease. More information can be found here https://archive.ics.uci.edu/dataset/45/heart+disease

## Exploring the Data

The project starts with an exploration of the dataset. We look for basic correlations between various features and the target values (‘0’ indicating the patient does not have heart disease and ‘1’ indicating the patient has heart disease). Various methods are used to compare and contrast the data, including crosstab evaluations and comparative scatter plots to visually display relationships.

*Below are a sampling of these datasets*
![Data Analysis](https://github.com/lorenarms/Machine_Learning/blob/main/heart_disease_project/img/data_analysis.png "Data Analysis")

From this data exploration, we can see that some features will play a larger role in the model’s learning than others will. For instance, age as an indicator of the presence of heart disease seems to show minimal correlation. Likewise, sex will play a very minor role in predicting heart disease in this dataset due to the larger number of male patients over female patients. However, according to the graph in the lower-right quadrant (“Heart Disease Frequency Per Chest Pain Type”) we can see that as patients report higher levels of chest pain they are more likely to also have heart disease. 

## Training The Models

To answer our question about using machine learning to predict heart disease in patients, we will employ three different classification models:

* Logistic Regression
* K-Nearest Neighbor Classifier
* Random Forest Classifier

All three of these models can be found in the SciKit-Learn library. A random seed of “42” is used when training and testing to ensure adequate replication of results by other testers. Each model also uses a test set of 20% of the total data.

After fitting the three models, we obtain baseline accuracy scores as shown in the graph below:
![Accuracy Score](https://github.com/lorenarms/Machine_Learning/blob/main/heart_disease_project/img/accuracy_scores.png "Accuracy Score")

The baseline scores are as follows:
| Model | Accuracy Score (out of 1.0) |
|---|---|
| Logistic Regression |	0.8852459016393442 |
| KNN | 0.6885245901639344 |
| Random Forest | 0.8360655737704918 |

These scores indicate that our most accurate model is the Logistic Regression, followed closely by the Random Forest. We will focus on tuning these two models to try and improve accuracy.

## Tuning the Models

As tuning by hand can be tedious, we employed the use of RandomSearchCV and GridSearchCV to automate and speed up the process of finding the best parameters for each model.

For the Logistic Regression model we focused on the `C` parameter. We offered a selection from the `np.logspace( -4, 4)` with a total sample generation of twenty. More information can be found here https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression

With the RandomForestClassifier model a range of options was provided for parameters `n_estimators`, `max_depth`, `min_sample_split`, and `min_samples_leaf`. For both models, twenty iterations were used with a cross-validation of 5, totalling 100 fits per model. After testing, the following parameters were found to be ‘best’:

| Model | Best Params |
|---|---|
| Logistic Regression | {'solver': 'liblinear', 'C': 0.23357214690901212} |
| RandomForestClassifier | {'n_estimators': 210, 'min_samples_split': 4, 'min_samples_leaf': 19, 'max_depth': 3} |

The following scores were the result of the above parameter usage:
| Model | Baseline | RandomSearchCV |
|---|---|---|
| Logistic Regression | 0.8852459016393442 | 0.8852459016393442 |
| RandomForestClassifier | 0.8360655737704918 | 0.8688524590163934 |

From the data above we can see that, even out of the box the Logistic Regression model does well during scoring. However, the RandomForestClassifier gained ground when using the best parameters selected by the RandomSearchCV. 

Because GridSearchCV is exhaustive we will focus on the Logistic Regression model solely for this part of the experiment. Using the same parameter options as above, we get:
| Baseline | RandomSearchCV | GridSearchCV |
|---|---|---|
| 0.8852459016393442 | 0.8852459016393442 | 0.8852459016393442 |

With no discernable difference between scores, we can draw one of two possible conclusions:

1. The model performs very well with the default parameters, with no extra tuning necessary
2. The model requires more fine-tuning through other parameters

## Evaluation Beyond Accuracy

There are a number of other values to look at when considering the success of a machine learning model, including the ROC Curve and AUC Score, and the Confusion Matrix of the predictions the model makes. 

As shown in the below figure, the ROC Curve indicates a much better consistency in correct prediction rates than simply guessing. The curve also has an AUC Score of 0.93.

![ROC](https://github.com/lorenarms/Machine_Learning/blob/main/heart_disease_project/img/roc_log_reg.png "ROC")

When evaluating the model's effectiveness of predicting whether a patient has heart disease or not we must also consider the false positive and false negative predictions. In the below figure the Confusion Matrix compares these, showing that out of 61 total test samples, 4 were assigned a “True” when the correct label was “False”, while the opposite was true for 3 of the samples. However, this amount is still consistent with the accuracy score of 88.5%.

![Confusion](https://github.com/lorenarms/Machine_Learning/blob/main/heart_disease_project/img/confusion_matrix.png "Confusion")

Finally, it is important to consider all features that the data provides to the model, and whether some of those features are not necessary to predictions. After fitting the model we can calculate the importance of each feature and visually show this in a figure (below). Features with higher numbers are more important to the model when making predictions, while features with lower numbers are less important, or even misleading to the model.

![Features](https://github.com/lorenarms/Machine_Learning/blob/main/heart_disease_project/img/feature_importance.png "Features")

The above graphic is consistent with our original interpretation of the data in that cp (chest pain) would prove to be of more importance when predicting heart disease in patients, while sex was not as important. 

## Conclusion

Having not hit the evaluation metric of 95% with our model, there are some things to consider:
1. More data may be necessary to train the model adequately. The current dataset contained 303 data points, which may have contributed to the lower accuracy score.
2. A more robust model may be needed, as there are several other classification models in existence that we did not try yet.
3. The current model could potentially be tuned more. The RandomForestClassifier, for instance, was not run through GridSearchCV, and thus a “best params” value was not found.

However, the experiment does show the beginnings of a “proof of concept”, in that a machine learning model was able to be trained to predict whether a patient does or does not have heart disease, based solely on a few features. A more in depth experiment is worth exploring in the future.

## Contact

<p>Check out my <a href="https://www.youtube.com/channel/UCGtp8PRHgPCQHYoSxbMST8A" target="_blank">YouTube channel</a> for more videos about coding projects I've done.</p>
<p>Also, check out my <a href="http://artllj.com" target="_blank">Personal Website</a> for more information about me, and my <a href="https://www.linkedin.com/in/lorenarms95/" target="_blank">LinkedIn</a> to see if I'd be a good fit for your team. </p>
<h3>Thanks for stopping by!</h3>
<img src="https://github.com/lorenarms/SNHU_CS_370_Emerging_Trends_in_CS/blob/main/images/profile.png" alt="[picture of me]" style="width:100px;">
<p>much love
-L
</p>
