# Predicting Bulldozer Sale Prices Using Machine Learning
<a href="https://github.com/lorenarms/Machine_Learning"><strong> ⇦ Go Back</strong></a>

The goal of this project was to demonstrate that, given enough data and features within a set, an machine learning model could be developed to predict the sale-price of a bulldozer with a 95% accuracy rate.

## The Data Source

The data sets are provided by Kaggle. The data was for a contest designed to let machine learning enthusiasts show their skills in developing models. The data comes in three main structured files: a ‘train’ set, a ‘validation’ set, and a ‘test’ set, the latter missing the ‘sale price’ data feature. The training set contains over 40,000 entries. Information on the data and the contest can be found at https://www.kaggle.com/c/bluebook-for-bulldozers

## Exploring the Data

As with any new data set, the goal is to explore the data, looking for any patterns that could give clues as to how the machine will learn the patterns. 

*Some samples of the data exploration are found below.*
![Exploring the Data](https://github.com/lorenarms/Machine_Learning/blob/main/bulldozer_price_prediction_project/img/explore.png "Exploring the Data")

In the above figure, the correlation between bulldozer sale month and the sale price is explored, as well as the correlation between sale year and sale price. Based on these graphs (which shows the mean price for the given x-coordinate values), we can see a pattern emerging that shows seasonal fluctuations in bulldozer sale prices. 

Other data explorations reveal similar patterns when comparing the various features of the bulldozers and their respective sale prices. 

## Shaping the Data

As with any dataset there are values that are missing, and others that are in non-integer or boolean values. To rectify this we’ll employ a function to first fill any missing numeric values with the median from the column:

`# fill numeric rows with median
	for label, content in df.items():
    	if pd.api.types.is_numeric_dtype(content):
        	if pd.isnull(content).sum():
            	# add a binary column which tells if the
            	# data was missing
            	df[label+"_is_missing"] = pd.isnull(content)
              #fill missing numeric values with median
            	df[label] = content.fillna(content.median())`

We use the median to ensure that any major outliers do not skew the data. Likewise, a new column to track missing data values is added to the end of the dataset.

Next we will fill any categorical values that are missing by first converting them to numbers, and then adding a similar tracking column as before.             	
    
    	`# fill categorical missing data and turn categories into numbers\
    	if not pd.api.types.is_numeric_dtype(content):
        	df[label+"_is_missing"] = pd.isnull(content)
        	df[label] = pd.Categorical(content).codes+1`

Some tracking columns are missing in the validation set due to certain features not missing any values, so we force-add these columns to the end with ‘false’ as the default value (indicating that these values were not missing in the first place).

The data is now shaped appropriately and the model will be able to read it and ultimately learn from it.

## Training the Model

Because the predictions we want to make are not classifications we will employ the RandomForestRegressor from `sklearn.ensemble`. To cut down on training time we’ll set the `max_samples` to 10,000 entries. 

The following metrics are generated from the training and validation sets:
| Metric | Value |
|---|---|
| Training Mean Abs Error | 5548.337219520101 |
| Validation Mean Abs Error | 9870.13091506092 |
| Training RMSLE | 0.25729129016365265 |
| Validation RMSLE | 0.3945193505788297 |

After training with the `train` dataset and validating results on the `valid` dataset we get the following R^2 values (accuaracy):

| Metric | Value |
|---|---|
| Training R^2 | 0.8611282073934683 |
| Validation R^2 | 0.6558038240154331 |

## Evaluation

Because we do not have the final sales data for the `test` dataset, we can instead use the above sets to compare the accuracy of the model. We can take the predictions the model made on the validation set and compare them to the actual data from the validation set. We’ll then take the mean of the differences:

| Value |
|---|
| +/- 3727.3270785448885 |

While this score is somewhat acceptable, we can likely tune the model to generate better scores overall. 

## Tuning the Model

Due to the large data size, we’ll focus on RandomizedSearchCV as opposed to GridSearchCV. GridSearchCV can be used in future projects to dial in the best parameters to use. 

Using RandomizedSearchCV and the following distributions:

`grid = {"n_estimators": np.arange(10, 100, 10),
   	"max_depth": [None, 3, 5, 10],
   	"min_samples_split": np.arange(2, 20, 2),
   	"min_samples_leaf": np.arange(1, 20, 2),
   	"max_features": [0.5, 1, "sqrt", None],
   	"max_samples": [25000]}`

…we can obtain a higher training accuracy:

| Metric | Value |
|---|---|
| Training R^2 | 0.8765075966051655 |
| Validation R^2 | 0.640142636897514 |

This did, however, show a drop in validation accuracy, meaning more testing needs to be done in order to increase overall model accuracy. 

The evaluation metric of the contest was the RMSLE (root mean squared log error) between the actual and predicted auction prices. The lower the value, the better. In our case, we obtained a RMSLE value of 0.3925351947398932 on our validation set. 

## Feature Importance

Finally, we want to explore the features of the data and see which hold the most importance for the model to learn from. We can plot the top twenty most important features in a chart, descending from most to least important:

![Feature Importance](https://github.com/lorenarms/Machine_Learning/blob/main/bulldozer_price_prediction_project/img/importance.png "Feature Importance")

As is shown above, the year the bulldozer is made and its size are the most determinant in predicting the sale price of the bulldozer. 

## Conclusion

The lowest RMSLE scores posted for the contest are 0.29. Having not met that mark implies the model needs more training. Likely removing some features of the data that are not needed, as well as implementing a GridSearchCV distribution to exhaust all possible parameters could lead to a much better score. As with other machine learning models, the project does show promise in its ability to predict the sale price of bulldozers. With a dataset of 40,000+ entries, it is likely the model itself that is underperforming, and better parameters need to be implemented to obtain a lower score.

<a href="https://github.com/lorenarms/Machine_Learning"><strong> ⇦ Go Back</strong></a>

## Contact

<p>Check out my <a href="https://www.youtube.com/channel/UCGtp8PRHgPCQHYoSxbMST8A" target="_blank">YouTube channel</a> for more videos about coding projects I've done.</p>
<p>Also, check out my <a href="http://artllj.com" target="_blank">Personal Website</a> for more information about me, and my <a href="https://www.linkedin.com/in/lorenarms95/" target="_blank">LinkedIn</a> to see if I'd be a good fit for your team. </p>
<h3>Thanks for stopping by!</h3>
<img src="https://github.com/lorenarms/SNHU_CS_370_Emerging_Trends_in_CS/blob/main/images/profile.png" alt="[picture of me]" style="width:100px;">
<p>much love
-L
</p>
