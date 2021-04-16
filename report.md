## Modeling Workflow

### Initial EDA

Our first task was to explore the dataset and identify any relevant characteristics that would inform our modeling techniques. Among our findings were two important characteristics. The first was the clear class imbalance:

![](images/class_breakdown.png)

Only about 9% of the total observations in our dataset were classified as fraud. In order to ensure our models are actually working to identify fraud, this imbalance needed to be taken into account, perhaps by passing class weights to our models. Furthermore, it meant that we would have to consider metrics other than pure accuracy to properly evaluate our model, since a model that only ever predicts 'not fraud' would still have a relatively high accuracy of 91%. In particular, we want to look at scores like precision, recall, roc-auc, and f-1. 

Additionally, we noted that there were a combination of numerical, categorical, and natural language features in the data. In order to wring as much predictive value out of our data as possible, we decided to split up the work into two parallel modeling groups. One group would build a model focusing purely on the numerical and categorical data, while the other would build a model using natural language processing. Once those models were complete, we could use a combination of their predicted probabilities for a given test example to make a final prediction.

### Preprocessing the Numerical and Categorical Data

Several important steps were taken to process the numerical/categorical data before modeling:
 - Removing natural language features
 - Identifying and one-hot encoding the necessary categorical features. Ensuring this was done in such a way as to be reproducible when testing on new data.
 - Locating and filling nans with reasonable substitutes
 - Extracting numerical data from 'object' columns (e.g. 'previous_payouts') and feature engineering (e.g. latitude/longitude becoming a boolean 'has_lat_long')

Obviously, it was important to preserve all of these steps as a preprocessing pipeline for use during the testing/production phase

### Metrics selection

As mentioned previously, the task is to identify fraud cases, but with fraud being a relatively infrequent class, alternatives to accuracy must be used. In particular, using an ROC-AUC score and visualizing with the ROC curve will be useful in maximizing our true positive identifications of fraud and minimizing our false positive rate. 
