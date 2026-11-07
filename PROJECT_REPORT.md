# HouseLens — Project and Technical Explanation

## Project purpose

HouseLens is a local Streamlit prototype for exploring Dutch homes and housing prices. A visitor can search an address, inspect public building and parcel facts, browse available listing data, and see how a small machine-learning model responds to property features. A separate Ames, Iowa demo shows how the original regression exercise was trained and evaluated.

The app is a learning and showcase project. Its Dutch estimate is not a professional valuation, a current asking price, or a guarantee of a future sale price.

## Tools and technologies

| Tool | How HouseLens uses it |
| --- | --- |
| Python | Main programming language for the interface, data handling, model training, and API calls. |
| Streamlit | Builds the local website, forms, navigation, metrics, tabs, tables, and interactive controls. The landing page uses a small amount of custom HTML and CSS for its visual layout. |
| pandas | Reads and cleans CSV, JSON, and CBS data; prepares model inputs; filters listing records; and combines municipality prices with price-index data. |
| scikit-learn | Provides the regression models, preprocessing pipelines, missing-value handling, train/test split, and evaluation metrics. |
| joblib | Saves and reloads the trained Ames model as a `.joblib` artifact. |
| Matplotlib | Draws the historical Ames price chart and model-result visuals. |
| Streamlit charts | Displays the recent CBS municipal price-index trend beside the market-update figure. |
| PyDeck | Displays the searched property's map point and, when available, the cadastral parcel outline. |
| Python standard library | `urllib` and `json` make HTTP requests and process API responses; `pathlib` locates project files; `re` and `datetime` support text matching and scheduled feed refreshes. |
| Jupyter Notebook | `HouseLens_training.ipynb` documents the data inspection, model comparison, error chart, prediction review, and separate Kaggle test-file steps. |
| VS Code and Streamlit | Used to edit and run the project locally with `streamlit run app.py`. |

The app does not use a generative AI model to price houses. “AI estimate” refers to a scikit-learn Random Forest model. The separate Tavily connection can search the web when a key is configured; those search results are not used to train the valuation model.

## Data sources and files

### Dutch housing information

- **PDOK Locatieserver** searches registered Dutch addresses.
- **BAG (Basisregistratie Adressen en Gebouwen)** supplies the registered dwelling floor area, use, status, and linked building record.
- **BRK / cadastral data** supplies parcel details and geometry when it can be matched to the address.
- **3DBAG** can add modelled building facts such as floors, volume, and footprint.
- **CBS StatLine** supplies the latest annual average sale price for existing homes by municipality.
- **CBS and Kadaster's municipal PBK index** updates the annual CBS price baseline to the latest available municipal quarter. HouseLens also plots recent quarterly values from this same index.
- **Klavida's open feed** can supply permitted active sale listings. The app refreshes and caches this feed no more than once per hour. The feed currently does not provide listing photos, and it does not provide a rental feed.
- **`data/listings.csv`** contains listing records stored locally, including a manually entered Merelhoven 343 listing. Its €495,000 figure is the asking price in that record; it is not the model's sale-price target.
- **`data/sale_calibrations.csv`** records the user-provided statement that Merelhoven 343 sold for more than €515,000. The exact amount and date are unknown, so HouseLens treats it as a lower-bound reference for that address only. It is not an additional machine-learning training row and does not change other properties' estimates.

Public registries describe buildings; they do not establish that a home is currently available, provide room counts, or supply listing photos. Listing information and registry facts are therefore presented as different kinds of evidence.

### Ames model files

- **`data/train.csv`** contains 1,460 labeled Ames, Iowa property records and the `SalePrice` target.
- **`data/test.csv`** contains 1,459 records without known sale prices. It can receive predictions, but cannot be used to score accuracy.
- **`artifacts/house_price_model.joblib`** stores the selected Ames model.
- **`artifacts/model_info.json`** stores details about the split, model choice, and reported score.
- **`reports/model_results.csv`**, **`reports/price_trend.csv`**, and **`reports/test_predictions.csv`** store evaluation, historical chart, and Kaggle-format output data.
- **`utrechthousingsmall.csv`** is a separate 100-row Dutch sample from four postcode areas in Utrecht and Amersfoort. It contains the `retailvalue` target used by the experimental Dutch feature model. It has no Capelle-specific sales. The app does not use a `taxvalue` column.

## How a Dutch address estimate works

1. A user enters an address on the landing page. HouseLens carries the address and Buy/Rent choice into the existing housing view and starts the address search.
2. PDOK returns matching registered addresses. After the user selects one, HouseLens requests the corresponding BAG dwelling, linked BAG building, cadastral parcel, and optional 3DBAG facts.
3. HouseLens matches the municipality code to the CBS annual average sale price for existing homes.
4. It updates that average using the municipality's CBS/Kadaster price-index change from the base year to the latest published quarter. Both the numeric market update and a small quarterly trend graph use the same index series.
5. A Random Forest estimates a property-feature multiplier using the local Dutch sample. Its training sale prices are normalized within each sample postcode area, so postcode price differences are not treated as house features. The model uses available facts such as floor area, parcel area, and build year; missing details are filled using training medians.
6. The feature multiplier is applied to the indexed CBS municipality baseline. HouseLens shows the market baseline, market change, trend, and model explanation separately.
7. For Merelhoven 343, HouseLens additionally displays the user-provided sale-price lower bound. It is explicitly labelled as sale-informed information, not an independently predicted exact value.

In simplified form:

```text
Dutch model estimate = CBS municipality average
                       × local CBS/Kadaster price-index adjustment
                       × Random Forest property-feature multiplier
```

The current Dutch sample is small and comes from different municipalities. The CBS average is for a whole municipality, not a set of comparable sales for one street. As a result, the Dutch model is useful for demonstrating the workflow, but it cannot promise the sale price for an individual home.

## How the Ames machine-learning exercise works

The training script is `train_model.py`. It uses eight fields: overall quality, above-ground living area, garage capacity, basement area, full bathrooms, build year, lot size, and neighborhood. `SalePrice` is the value to predict.

The script:

1. Splits the labeled data into 80% training and 20% holdout records using a fixed random seed.
2. Fills missing numeric values with the training median, fills a missing neighborhood with the most common category, and converts neighborhood names using one-hot encoding.
3. Trains and compares a median-price baseline, linear regression, and Random Forest regression.
4. Compares the models using mean absolute error (MAE), root mean squared error (RMSE), and R-squared (R²). The best model by holdout MAE is then retrained on all labeled Ames records and saved with joblib.
5. Writes predictions for the separate unlabeled Kaggle test file. Those predictions cannot be evaluated without its true sale prices.

In the current saved report, the Random Forest holdout MAE is about $18,530. This score describes the Ames holdout split only. It is not an accuracy score for Dutch homes and should not be used to describe the Dutch estimate's error.

## What the website sections do

- **Landing page:** Introduces HouseLens, lets a visitor choose buying or renting, and starts an address search.
- **Browse homes:** Shows registered property details, CBS-based sale context, the municipal trend graph, the permitted sale feed, and locally stored listing cards. Rental browsing is present as an interface choice, but no rental provider feed or rent model is connected.
- **Price estimate demo:** Lets the user choose a municipality and enter property characteristics to see the Dutch model's demonstration estimate and feature sensitivities.
- **Past prices demo:** Uses the historical Ames data and presents prices in US dollars.
- **Model results demo:** Shows the Ames model comparison and feature importance.
- **Web search:** Uses Tavily when `TAVILY_API_KEY` is configured. It is a separate information-search feature, not an input to the model.

## Important limitations

- The Ames training data is historical and from one US city; its model and score do not validate Dutch predictions.
- The Dutch model has only 100 sample records from four postcode areas and no Capelle-specific transaction sample.
- CBS municipality prices are averages. The index updates overall market movement, but does not account for a particular home's renovations, condition, view, or exact street-level comparables.
- The user-reported Merelhoven sale amount is a lower bound with no date or exact amount. It is shown only as a reference for that property.
- The listing feed has no photos or rentals at present. Photos are shown only when a permitted listing source supplies them.
- Model feature sensitivities show how the model responds to inputs; they do not prove that a feature causes a price change.
- Web results can be incomplete or out of date. They are shown with their source links and do not silently modify model predictions.
- There is no pinned dependency file in the project, so a fresh installation may require installing the imported Python packages separately.

## Short presentation summary

HouseLens is a local Dutch housing explorer built with Python and Streamlit. It combines public address and building records from PDOK, BAG, BRK, and 3DBAG with municipality-level sale prices and market trends from CBS and Kadaster. A small scikit-learn Random Forest adjusts the local market baseline using property features. The site also includes listing browsing, a separate web-search tool, and an Ames, Iowa model-training demonstration. The Dutch estimate is a prototype: it has limited local training data and is not a formal valuation.
