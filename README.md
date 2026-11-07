# HouseLens

HouseLens is a local Streamlit project for exploring Dutch homes, public property records, and housing-price context. It combines an address lookup, a small Dutch property-feature model, CBS market data, a permitted sale-listing feed, and a separate Ames, Iowa model-training demonstration.

> **Important:** The Dutch price output is an educational prototype, not a professional valuation. The Dutch model has a small Utrecht/Amersfoort sample and no local transaction training set for most municipalities. The Ames model and its accuracy score are for Iowa data only.

## What the website does

- Opens on a landing page where a visitor chooses **Buy** or **Rent** and can start an address search.
- Looks up registered Dutch addresses and shows available BAG building facts, cadastral parcel information, and optional 3D building facts.
- Uses CBS municipality-level average sale prices as a location baseline and updates that baseline with the latest available CBS/Kadaster municipal price index.
- Shows a recent quarterly price-index chart beside the market update.
- Applies a Random Forest feature multiplier to the indexed CBS baseline and explains which available property facts move the model result.
- Browses current sale listings from Klavida's permitted open feed and local listing records. Photos appear only when a source supplies a usable photo URL.
- Provides an Ames, Iowa estimate, historical price chart, and model-comparison view as a separate machine-learning demonstration.
- Offers optional web search through Tavily when an API key is configured.

The interface includes a rent option, but HouseLens currently has no connected rental feed or rent-estimation model.

## Technology used

| Technology | Role |
| --- | --- |
| Python | Main programming language and application logic. |
| Streamlit | Local website, landing page, forms, tabs, navigation, metrics, and charts. |
| HTML and CSS | Styling for the landing-page hero and navigation. |
| pandas | Reading and cleaning CSV and JSON data, transforming property features, and combining CBS price data. |
| scikit-learn | Random Forest regression, preprocessing, missing-value handling, model comparison, and evaluation. |
| joblib | Saving and loading the Ames model artifact. |
| Matplotlib | Historical Ames charts and model-result plots. |
| Streamlit charts | Compact quarterly CBS municipal price-index trend. |
| PyDeck | Map point and cadastral parcel display. |
| Python standard library (`urllib`, `json`, `pathlib`, `datetime`, `re`) | Calling public web APIs, handling JSON responses, locating files, formatting dates, and matching address text. |
| Jupyter Notebook | Step-by-step exploration and comparison of the Ames models in `HouseLens_training.ipynb`. |

HouseLens does not use a generative AI model to estimate prices. Its price estimates come from scikit-learn regression models. Tavily is a separate web-search feature; its results do not train or silently alter the valuation model.

## Run HouseLens locally

The project is intended to run locally. In PowerShell, from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The saved Ames model is included in `artifacts/house_price_model.joblib`, so the app does not need to be trained before it starts. To rerun the Ames training and regenerate its outputs:

```powershell
python train_model.py
```

### Optional web search setup

Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml` and put your Tavily API key in the new file:

```toml
TAVILY_API_KEY = "your-key"
```

The real `secrets.toml` is excluded from Git. Do not commit API keys. HouseLens's public-record, CBS, and Klavida features do not require a Tavily key.

## How the Dutch address estimate works

1. **Address search:** The landing page passes the address and Buy/Rent choice into the housing view. PDOK's Locatieserver returns matching registered addresses.
2. **Building details:** HouseLens requests the selected dwelling from BAG, follows its linked building record, and looks up a connected cadastral parcel in BRK. If available, 3DBAG adds modelled building facts such as floors, volume, and footprint.
3. **Local market baseline:** The address's municipality code is matched to CBS's latest annual average sale price for existing homes.
4. **Market update:** The CBS/Kadaster municipal PBK price index updates the annual average to the latest available quarter. The app displays both the percentage change and a trend chart from the same quarterly index series.
5. **Property features:** A Random Forest trained on `utrechthousingsmall.csv` estimates a multiplier from facts such as living area, plot area, build year, garden, bathrooms, energy code, and monument status. Missing address details are filled with training-data medians.
6. **Final model estimate:** The property-feature multiplier is multiplied by the indexed municipality baseline.
7. **Known sale reference:** `data/sale_calibrations.csv` contains a user-provided lower bound for Merelhoven 343. HouseLens shows this only for that matching address and labels it as sale-informed information, not an independent AI prediction.

In simplified form:

```text
Dutch model estimate
  = CBS municipality average
  × CBS/Kadaster market-index adjustment
  × Random Forest property-feature multiplier
```

### Dutch model training data

`utrechthousingsmall.csv` has 100 records from four Utrecht and Amersfoort postcode areas. Its target is `retailvalue`. During training, target prices are normalized against the median for each sample postcode area so that the model focuses on property-feature differences rather than using the postcode itself as a location price. The model does not use tax values, IDs, coordinates, or the undefined `select` field.

This sample is too small and geographically narrow to produce a dependable address-level Dutch valuation. CBS provides a current municipality baseline, but it is an average across many kinds of homes—not a set of comparable sales on the same street. HouseLens clearly separates the local market benchmark, model estimate, and known sale reference for that reason.

## The separate Ames, Iowa model

`train_model.py` trains the original machine-learning exercise using `data/train.csv`, which contains 1,460 Ames properties with known `SalePrice` values. It uses eight features: overall quality, above-ground living area, garage capacity, basement area, full bathrooms, build year, lot size, and neighborhood.

The training script:

1. Holds out 20% of labeled records for evaluation using a fixed random seed.
2. Fills missing numeric values with the training median and missing neighborhood values with the most common neighborhood.
3. One-hot encodes neighborhood and compares a median-price baseline, linear regression, and Random Forest regression.
4. Scores the models with MAE, RMSE, and R². The best model by holdout MAE is retrained on all labeled rows and saved with joblib.
5. Generates predictions for `data/test.csv`, which has 1,459 records but no known sale prices. Those predictions cannot be scored without the missing labels.

The saved Ames model's reported holdout MAE is about **$18,530**. That figure describes a held-out sample of historical Ames sales only; it is not a Dutch-model error estimate.

## External data and integrations

| Source | HouseLens use |
| --- | --- |
| PDOK Locatieserver | Nationwide search for registered addresses. |
| BAG | Registered dwelling area, status, use, and linked building record. |
| BRK / cadastral service | Connected parcel facts and map geometry where available. |
| 3DBAG | Optional 3D building model properties. |
| CBS StatLine | Latest annual municipality average sale prices for existing homes. |
| CBS and Kadaster municipal PBK | Municipal quarterly index used for market updates and the trend graph. |
| Klavida open feed | Permitted sale-listing feed, refreshed no more than once per hour and cached locally. It currently supplies no photos or rental listings. |
| Tavily Search API | Optional web-search results, shown with source links. Requires `TAVILY_API_KEY`. |

HouseLens does not scrape Funda or use Street View. The local Merelhoven listing is a manually entered record in `data/listings.csv`; the €495,000 there is the listing's asking price. The user-reported sale reference above €515,000 is stored separately as a lower bound because the exact amount and sale date were not provided.

## Project files

```text
app.py                          Streamlit website and Dutch address-estimate logic
train_model.py                  Ames model comparison and artifact generation
HouseLens_training.ipynb        Notebook walkthrough of the Ames model exercise
PROJECT_REPORT.md               Detailed project and technical report
requirements.txt                Python package versions used by the project
utrechthousingsmall.csv         Small Dutch property-feature training sample
houselens_listings_template.csv Listing CSV column template
data/listings.csv               Locally stored listing records
data/sale_calibrations.csv      User-provided property-specific sale lower bound
data/train.csv                  Labeled Ames training data
data/test.csv                   Unlabeled Kaggle test data
artifacts/                      Saved Ames model and model metadata
reports/                        Model scores, historical chart data, and predictions
.streamlit/config.toml          Streamlit color theme
.streamlit/secrets.example.toml Example configuration without real secrets
```

## Limitations and responsible use

- The Ames model is trained on historical U.S. data and is separate from the Dutch estimate.
- The Dutch feature model uses only 100 records from four postcode areas and has no Capelle-specific transaction training sample.
- A municipality-wide average and price index cannot account for a particular home's renovation quality, condition, exact street, or recent comparable sales.
- The Merelhoven sale reference is user-provided, date-unknown, and only a lower bound. It is not used as a general multiplier or training label.
- The sale feed currently has no listing photos, and no rental provider or rent model is connected.
- Feature explanations describe model sensitivity; they do not prove that a feature causes a price change.
- Web-search results may be incomplete or outdated and do not modify estimates automatically.
- The website is a local prototype, not a production service or financial advice.
