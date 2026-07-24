# Model Card

## Model Details

This model is a histogram-based gradien boosting regression model (more info here: 
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).Its hyperparameters, which 
can be found in the `train_model` function in `/ml/model.py`, are currently as follows:

* loss='absolute_error'
* random_state=42
* learning_rate=0.4
* max_iter=550
* max_leaf_nodes=125
* max_depth=75
* min_samples_leaf=75

These hyperparams were determined based on a grid-search using mean_absolute_error as its scoring, set with 
`scoring='neg_mean_absolute_error'`.


## Intended Use 

This model is intended to be used to predict the price of a home based on a small set of features and its location. Such features 
can be found in the README.md in the "Input" section. 


## Training Data

The data used to train this data is publicly available on Kaggle and can be found here: 
https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset. Specifically, 75% of the linked data was used for 
training.


## Evaluation Data

The data used to evaluate this model is the left over 25% of the data linked above.


## Metrics

The metrics for the model can be found at "/model/metrics.json", or by running the train_model.py script, as the metrics are 
printed to the terminal when training the model. Currently, they are as follows:

* Mean Absolute Error:  74,506.49
* Median Absolute Error: 44,616.62
* $$R^2$$ Score: 0.7452


## Ethical Considerations

Theoretically, this model could be used by people to determine how they should evaluate a house. Sellers could take the maximum 
estimate and claim that, since an AI model predicted that estimate, that value is mathematically correct, and refuse to accept 
lower offers. Similarly, buyers could use the model, and not accept any value above the minimum estimate. Finally, buyers and 
sellers could discriminate against certain individuals, using the high estimate for some, but the low estimate for others, then 
claim that they are using the same model for everyone, therefore they aren't discriminating.

Technically, I could only output the prediction, and not the range, but I would also, in that case, have to either not publicize 
the error metrics or publicize them, but try to make it seem like they don't have a significant impact on the accuracy of the 
predictions. Both of these options involve obscuring the models performance in some way, however, which is unethical in its own 
way.


## Caveats and Recommendations

### Caveats

It is important to note that the training dataset only contained a fairly small number of features. It did not include, and 
therefore the model cannot predict based on, the number of floors, building materials, construction time, whether or not the house 
is located in a particularly high-value location, as may be the case for many beach or lake houses, when the house was built, etc. 
because of this, the model cannot explain all of the variance in price, and therefore may be wildly inaccurate at times. A second 
caveat is that this data was uploaded to in 2024, and if we assume that is the date of the data itself, that means the data is at 
least two years old now (as of July 2026). Therefore, the relevance of the data has already deteriorated somewhat.

Another major caveat is a simple economic priciple: things do not have an intrinsic monetary value; instead, the "cost" in currency 
of an item, such as a home, is entirely dependent on what the seller believes it to be. This is another variable that the model 
cannot take into account, though ideally, this variable does not represent a major portion of the variance in price.


### Recommendations

To ensure accuracy of the model when running, you should include as much information about the house you are trying to predict the 
price of. In other words, try to use as many of the columns as you can, but don't feel pressured to use every column.


