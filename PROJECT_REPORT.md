# HouseLens project report

## Project purpose

HouseLens explores whether a small machine-learning model can estimate the sale price of a house from a few property details. A simple website lets a user change the details, view the estimate, and compare it with historical area prices. The aim is to support comparison and learning, not decide whether someone should buy a house.

## Research question

How accurately can beginner-friendly regression models estimate historical Ames house sale prices from selected property characteristics, and how should their limitations be shown to a website user?

## Data

The project uses the supplied Kaggle House Prices: Advanced Regression Techniques files. The labeled `train.csv` file contains 1,460 properties and actual `SalePrice` values. The separate `test.csv` file contains 1,459 properties but no sale-price labels. The houses are in Ames, Iowa, and the model outputs historical US dollars.

Eight input features are used: overall quality, above-ground living area, garage capacity, basement area, full bathrooms, year built, lot size, and neighborhood. Missing numeric values are filled with the training median. Missing neighborhood values are filled with the most common neighborhood, and neighborhood names are converted to numeric indicators by one-hot encoding.

## Method

The data is split once into 80% for training and 20% for evaluation, with a fixed random seed so the exercise can be repeated. Three models are compared: a median-price baseline, linear regression, and a random forest. The metrics are mean absolute error (MAE), root mean squared error (RMSE), and R-squared (R2). Lower MAE/RMSE is better; R2 closer to one indicates a better fit than a simple average-based prediction.

After the comparison, the best model by holdout MAE is retrained using all 1,460 labeled rows for the website. The unlabeled Kaggle test file is used only to produce a prediction CSV. Its predictions cannot be scored without actual sale prices.

## Results

| Model | Holdout MAE (USD) | Holdout RMSE (USD) | R2 |
| --- | ---: | ---: | ---: |
| Random forest | 18,529.55 | 29,695.05 | 0.8850 |
| Linear regression | 21,842.99 | 36,228.29 | 0.8289 |
| Median price baseline | 59,568.25 | 88,667.17 | -0.0250 |

On this one split, the random forest had the smallest average absolute error. The result is useful evidence for this prototype, but it does not show that the model will work equally well on current sales, other cities, or unusual houses. The random forest's average error is shown as a rough reference in the website; it is not a guaranteed bound or a formal prediction interval.

## Website

The Streamlit website has three sections. The estimate form uses eight user-selected inputs. A historical chart shows median sale prices by Ames neighborhood and year. A results section shows the comparison metrics and which groups of features mattered to the random forest. The interface labels the data location, currency, time period, and uncertainty, and says that the result is not financial advice.

## Responsible-AI choices

1. The app describes its output as an estimate and leaves the decision with the user.
2. It shows the comparison metrics and a simple baseline, so model performance is not presented without context.
3. It identifies the dataset as historical Ames, Iowa sales rather than implying Dutch market coverage.
4. It collects no names, accounts, or stored user preferences.
5. It describes the displayed average error as a rough reference, not a certainty range.
6. It shows historical trend data separately from the model's prediction and warns that a short historical period is not a forecast.

## Limitations and next steps

The dataset is small, historical, and from one US city. The model omits many details, including renovation quality, exact condition, current interest rates, and current supply and demand. One random split can produce a different result from another split. Neighborhood coverage and transaction counts are uneven. Feature importance describes how the model used input fields; it does not show cause and effect. The price chart only covers sales from 2006 to 2010.

Useful next steps would be to ask potential users whether the explanation is clear, compare results over repeated splits or cross-validation, inspect large prediction errors, and source current, appropriately licensed Dutch data before making claims about the Netherlands. A production system would need a fresh privacy, data-rights, fairness, and legal review.

## Conclusion

In the held-out sample, the random forest performed better than the linear model and median baseline on MAE. HouseLens turns that result into an interactive learning demo with visible context and limitations. The model's result supports further exploration; it is not enough to make reliable real-world house-buying recommendations.
