"""A small Streamlit demo for exploring the HouseLens model."""

from pathlib import Path
import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from io import BytesIO
from urllib.parse import quote, urlencode, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import pydeck as pdk
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).parent
MODEL_FILE = ROOT / "artifacts" / "house_price_model.joblib"
INFO_FILE = ROOT / "artifacts" / "model_info.json"
TREND_FILE = ROOT / "reports" / "price_trend.csv"
DATA_FILE = ROOT / "data" / "train.csv"
UTRECHT_TRAINING_FILE = ROOT / "utrechthousingsmall.csv"
LISTINGS_FILE = ROOT / "data" / "listings.csv"
SALE_CALIBRATIONS_FILE = ROOT / "data" / "sale_calibrations.csv"
KLAVIDA_FEED_URL = "https://klavida.nl/open/aanbod.json"
KLAVIDA_CACHE_FILE = ROOT / "data" / "klavida_open_cache.json"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
PDOK_LOCATIONS_URL = "https://api.pdok.nl/bzk/locatieserver/search/v3_1/free"
BAG_API_URL = "https://api.pdok.nl/kadaster/bag/ogc/v2"
BRK_API_URL = "https://api.pdok.nl/kadaster/brk-kadastrale-kaart/ogc/v1"
THREEDBAG_API_URL = "https://api.3dbag.nl/collections/pand/items"
CBS_MUNICIPAL_PRICE_API = "https://opendata.cbs.nl/ODataApi/odata/83625NED"
CBS_MUNICIPAL_PRICE_PAGE = "https://www.cbs.nl/nl-nl/cijfers/detail/83625NED"
CBS_MUNICIPAL_INDEX_PAGE = "https://www.cbs.nl/nl-nl/maatwerk/2026/33/prijsindex-bestaande-koopwoningen--pbk---naar-gemeente"
KLAVIDA_REFRESH_INTERVAL = timedelta(hours=1)
LISTING_COLUMNS = [
    "offer_type", "title", "city", "price", "area_m2", "rooms",
    "photo_url", "listing_url", "property_type", "energy_label", "source", "status", "description",
    "build_year", "lot_area_m2", "bedrooms", "bathrooms", "garden_area_m2", "solar_panels",
]
UTRECHT_DATA_COLUMNS = [
    "id", "zipcode", "lot-len", "lot-width", "lot-area", "house-area",
    "garden-size", "balcony", "x-coor", "y-coor", "buildyear", "bathrooms",
    "retailvalue", "energy-eff", "monument", "select",
]
UTRECHT_MODEL_FEATURES = [
    "lot-len", "lot-width", "lot-area", "house-area", "garden-size",
    "balcony", "buildyear", "bathrooms", "energy-eff", "monument",
]
UTRECHT_NUMERIC_FEATURES = UTRECHT_MODEL_FEATURES

st.set_page_config(page_title="HouseLens", page_icon="🏠", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_files():
    with open(INFO_FILE, encoding="utf-8") as f:
        info = json.load(f)
    return info, pd.read_csv(TREND_FILE), pd.read_csv(DATA_FILE)


@st.cache_data(ttl=900)
def load_listings_file(file_version=None):
    if LISTINGS_FILE.exists() and LISTINGS_FILE.stat().st_size > 0:
        return pd.read_csv(LISTINGS_FILE)
    return pd.DataFrame(columns=LISTING_COLUMNS)


def listing_photo_source(photo_url):
    """Resolve a provider image URL or a photo stored inside the HouseLens project."""
    if not isinstance(photo_url, str) or not photo_url.strip():
        return None
    photo_url = photo_url.strip()
    if photo_url.startswith(("https://", "http://")):
        return photo_url
    photo_path = Path(photo_url)
    if not photo_path.is_absolute():
        photo_path = ROOT / photo_path
    return photo_path if photo_path.is_file() else None


def normalize_utrecht_data(source):
    """Normalize Utrecht data, dropping any legacy taxvalue column."""
    frame = source.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    for column in UTRECHT_DATA_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    for column in UTRECHT_DATA_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["zipcode", "house-area", "retailvalue"])
    return frame[UTRECHT_DATA_COLUMNS].copy()


@st.cache_data
def load_utrecht_training_data(file_version=None):
    """Load the local Utrecht sample for model training, without showing its rows as listings."""
    if not UTRECHT_TRAINING_FILE.exists():
        return pd.DataFrame(columns=UTRECHT_DATA_COLUMNS)
    try:
        return normalize_utrecht_data(pd.read_csv(UTRECHT_TRAINING_FILE))
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=UTRECHT_DATA_COLUMNS)


def cbs_odata_items(url):
    items = []
    while url:
        payload = pdok_json(url)
        items.extend(payload.get("value", []))
        url = payload.get("odata.nextLink") or payload.get("@odata.nextLink")
    return items


def fetch_public_bytes(url):
    request = Request(url, headers={"User-Agent": "HouseLens local housing research"})
    with urlopen(request, timeout=30) as response:
        return response.read()


@st.cache_data(ttl=21600, max_entries=2, show_spinner=False)
def load_cbs_municipality_index_history():
    """Load the latest CBS/Kadaster quarterly index history for Dutch municipalities."""
    page_html = fetch_public_bytes(CBS_MUNICIPAL_INDEX_PAGE).decode("utf-8", errors="replace")
    workbook_links = re.findall(
        r'href=["\']([^"\']+\.xlsx(?:\?[^"\']*)?)["\']',
        page_html,
        flags=re.IGNORECASE,
    )
    versioned_workbooks = []
    for link in workbook_links:
        version = re.search(r"--(\d{4})q([1-4])\.xlsx", link, flags=re.IGNORECASE)
        if version:
            versioned_workbooks.append((int(version.group(1)), int(version.group(2)), link))
    if not versioned_workbooks:
        raise RuntimeError("CBS did not list a municipality price-index workbook.")

    latest_year, latest_quarter, workbook_url = max(versioned_workbooks)
    workbook = fetch_public_bytes(workbook_url)
    index_data = pd.read_excel(
        BytesIO(workbook),
        sheet_name="Tabel 1",
        header=None,
        skiprows=5,
        usecols=[0, 1, 2, 3],
        names=["period", "municipality_code_raw", "municipality", "price_index"],
    )
    index_data["municipality_code_raw"] = pd.to_numeric(
        index_data["municipality_code_raw"], errors="coerce"
    )
    index_data["price_index"] = pd.to_numeric(index_data["price_index"], errors="coerce")
    period_parts = index_data["period"].astype(str).str.extract(r"(?P<year>\d{4})\s+(?P<quarter>[1-4])e kwartaal")
    index_data["year"] = pd.to_numeric(period_parts["year"], errors="coerce")
    index_data["quarter"] = pd.to_numeric(period_parts["quarter"], errors="coerce")
    index_data = index_data.dropna(
        subset=["municipality_code_raw", "price_index", "year", "quarter"]
    ).copy()
    index_data["municipality_code"] = (
        "GM" + index_data["municipality_code_raw"].astype(int).astype(str).str.zfill(4)
    )
    index_data["period_label"] = (
        index_data["year"].astype(int).astype(str)
        + " Q"
        + index_data["quarter"].astype(int).astype(str)
    )
    return index_data[[
        "municipality_code", "municipality", "year", "quarter", "price_index", "period_label"
    ]].sort_values(["year", "quarter"]).reset_index(drop=True)


@st.cache_data(ttl=21600, max_entries=2, show_spinner=False)
def load_cbs_municipality_index_adjustments(base_year):
    """Calculate market updates from the CBS/Kadaster municipality PBK index."""
    index_data = load_cbs_municipality_index_history()
    if index_data.empty:
        return pd.DataFrame(columns=[
            "municipality_code", "market_adjustment_factor", "market_adjustment_pct", "market_index_period"
        ])
    latest = index_data.sort_values(["year", "quarter"]).iloc[-1]
    latest_year = int(latest["year"])
    latest_quarter = int(latest["quarter"])

    annual_base = (
        index_data[index_data["year"] == int(base_year)]
        .groupby("municipality_code", as_index=False)["price_index"]
        .mean()
        .rename(columns={"price_index": "base_year_index"})
    )
    current_period = index_data[
        (index_data["year"] == latest_year)
        & (index_data["quarter"] == latest_quarter)
    ][["municipality_code", "price_index"]].rename(
        columns={"price_index": "latest_price_index"}
    )
    adjustments = annual_base.merge(current_period, on="municipality_code", how="inner")
    adjustments["market_adjustment_factor"] = (
        adjustments["latest_price_index"] / adjustments["base_year_index"]
    )
    adjustments["market_adjustment_pct"] = (
        (adjustments["market_adjustment_factor"] - 1) * 100
    )
    adjustments["market_index_period"] = f"{latest_year} Q{latest_quarter}"
    return adjustments[[
        "municipality_code", "market_adjustment_factor", "market_adjustment_pct", "market_index_period"
    ]]


def cbs_municipality_price_trend(municipality_code, periods=20):
    """Return a compact recent CBS index series for a municipality."""
    if not isinstance(municipality_code, str) or not municipality_code:
        return pd.DataFrame(columns=["period_label", "price_index"])
    history = load_cbs_municipality_index_history()
    trend = history[history["municipality_code"] == municipality_code].tail(periods)
    return trend[["period_label", "price_index"]].reset_index(drop=True)


def show_cbs_municipality_price_trend(municipality_code):
    trend = cbs_municipality_price_trend(municipality_code)
    st.caption("Recent price trend · CBS index (2020 = 100)")
    if trend.empty:
        st.caption("Municipal trend data is not available for this area.")
        return
    st.line_chart(
        trend.set_index("period_label")["price_index"],
        height=115,
        use_container_width=True,
    )


@st.cache_data(ttl=21600, max_entries=2, show_spinner=False)
def load_cbs_municipality_prices():
    """Read the latest annual average sale prices published by CBS for municipalities."""
    periods_url = f"{CBS_MUNICIPAL_PRICE_API}/Perioden?{urlencode({'$select': 'Key,Title'})}"
    periods = cbs_odata_items(periods_url)
    annual_periods = [
        period for period in periods
        if str(period.get("Key", "")).endswith("JJ00") and str(period.get("Title", "")).isdigit()
    ]
    if not annual_periods:
        return pd.DataFrame(columns=[
            "municipality_code", "municipality", "average_price_eur", "year",
            "market_adjustment_factor", "market_adjustment_pct", "market_index_period",
            "market_adjusted_price_eur",
        ])
    latest_period = max(annual_periods, key=lambda period: period["Key"])
    period_key = latest_period["Key"]
    period_year = int(str(latest_period["Title"]))

    price_filter = f"Perioden eq '{period_key}' and startswith(RegioS,'GM')"
    prices_query = urlencode({
        '$filter': price_filter,
        '$select': 'GemiddeldeVerkoopprijs_1,Perioden,RegioS',
        '$top': '1000',
    })
    prices_url = f"{CBS_MUNICIPAL_PRICE_API}/TypedDataSet?{prices_query}"
    regions_query = urlencode({
        '$filter': "startswith(Key,'GM')",
        '$select': 'Key,Title',
        '$top': '1000',
    })
    regions_url = f"{CBS_MUNICIPAL_PRICE_API}/RegioS?{regions_query}"
    prices = pd.DataFrame(cbs_odata_items(prices_url))
    regions = pd.DataFrame(cbs_odata_items(regions_url))
    expected_columns = [
        "municipality_code", "municipality", "average_price_eur", "year",
        "market_adjustment_factor", "market_adjustment_pct", "market_index_period",
        "market_adjusted_price_eur",
    ]
    if prices.empty or regions.empty:
        return pd.DataFrame(columns=expected_columns)

    frame = prices.merge(regions, left_on="RegioS", right_on="Key", how="left")
    frame = frame.rename(columns={
        "RegioS": "municipality_code",
        "Title": "municipality",
        "GemiddeldeVerkoopprijs_1": "average_price_eur",
    })
    frame["average_price_eur"] = pd.to_numeric(frame["average_price_eur"], errors="coerce")
    frame = frame.dropna(subset=["municipality", "average_price_eur"])
    frame["year"] = period_year
    try:
        market_adjustments = load_cbs_municipality_index_adjustments(period_year)
        frame = frame.merge(market_adjustments, on="municipality_code", how="left")
    except Exception:
        frame["market_adjustment_factor"] = float("nan")
        frame["market_adjustment_pct"] = float("nan")
        frame["market_index_period"] = None
    frame["market_adjusted_price_eur"] = frame["average_price_eur"] * frame["market_adjustment_factor"].fillna(1)
    return frame[expected_columns].drop_duplicates("municipality_code").reset_index(drop=True)


def find_cbs_area_price(municipality_prices, municipality_code, municipality_name):
    if municipality_prices.empty:
        return None
    code = str(municipality_code or "").strip()
    if code:
        cbs_code = code if code.startswith("GM") else f"GM{code.zfill(4)}"
        match = municipality_prices[municipality_prices["municipality_code"] == cbs_code]
        if not match.empty:
            return match.iloc[0]

    search_name = str(municipality_name or "").strip().casefold()
    if search_name:
        names = municipality_prices["municipality"].fillna("").astype(str).str.strip().str.casefold()
        match = municipality_prices[names == search_name]
        if match.empty:
            match = municipality_prices[names.str.replace(" (gemeente)", "", regex=False) == search_name]
        if not match.empty:
            return match.iloc[0]
    return None


def load_sale_calibrations():
    """Load user-provided sale-price lower bounds for specific properties."""
    columns = ["street_address", "municipality", "sale_price_lower_bound_eur", "note"]
    if not SALE_CALIBRATIONS_FILE.exists():
        return pd.DataFrame(columns=columns)
    try:
        records = pd.read_csv(SALE_CALIBRATIONS_FILE)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in records:
            records[column] = pd.NA
    records["sale_price_lower_bound_eur"] = pd.to_numeric(
        records["sale_price_lower_bound_eur"], errors="coerce"
    )
    return records[columns].dropna(
        subset=["street_address", "municipality", "sale_price_lower_bound_eur"]
    )


def find_sale_calibration(address_label, sale_calibrations):
    """Match an exact street and municipality to a user-provided sale reference."""
    if not address_label or sale_calibrations.empty:
        return None

    normalized_address = re.sub(r"[^a-z0-9]+", " ", str(address_label).casefold()).strip()
    for _, record in sale_calibrations.iterrows():
        street = re.sub(r"[^a-z0-9]+", " ", str(record["street_address"]).casefold()).strip()
        municipality = re.sub(r"[^a-z0-9]+", " ", str(record["municipality"]).casefold()).strip()
        if street and municipality and street in normalized_address and municipality in normalized_address:
            return record
    return None


def build_utrecht_pipeline():
    return Pipeline(
        [
            ("prepare", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=6,
                    min_samples_leaf=3,
                    random_state=42,
                ),
            ),
        ]
    )


@st.cache_resource
def train_utrecht_model(dataset):
    """Fit within-postcode property-feature effects; postcode itself is not a model input."""
    training = normalize_utrecht_data(dataset).dropna(subset=["retailvalue"]).copy()
    postcode_medians = training.groupby("zipcode")["retailvalue"].transform("median")
    relative_values = training["retailvalue"] / postcode_medians
    estimator = build_utrecht_pipeline()
    estimator.fit(training[UTRECHT_MODEL_FEATURES], relative_values)
    return estimator, len(training)


def numeric_value_or_nan(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if pd.notna(result) else float("nan")


def predict_address_value(dwelling, building, parcel, training_data):
    """Predict a house-feature multiplier relative to a typical home in its area."""
    if training_data.empty or not dwelling:
        return None, None, None

    bag_props = dwelling.get("properties", {})
    building_props = (building or {}).get("properties", {})
    parcel_props = (parcel or {}).get("properties", {})
    features = pd.DataFrame(
        [{
            "lot-len": float("nan"),
            "lot-width": float("nan"),
            "lot-area": numeric_value_or_nan(parcel_props.get("kadastrale_grootte_waarde")),
            "house-area": numeric_value_or_nan(bag_props.get("oppervlakte")),
            "garden-size": float("nan"),
            "balcony": float("nan"),
            "buildyear": numeric_value_or_nan(building_props.get("bouwjaar")),
            "bathrooms": float("nan"),
            "energy-eff": float("nan"),
            "monument": float("nan"),
        }],
        columns=UTRECHT_MODEL_FEATURES,
    )
    estimator, _ = train_utrecht_model(training_data)
    feature_factor = float(estimator.predict(features)[0])
    return feature_factor, features, estimator


UTRECHT_FEATURE_LABELS = {
    "lot-len": "Lot length",
    "lot-width": "Lot width",
    "lot-area": "Parcel area",
    "house-area": "BAG floor area",
    "garden-size": "Garden size",
    "balcony": "Balcony count",
    "buildyear": "Build year",
    "bathrooms": "Bathrooms",
    "energy-eff": "Energy-efficiency code",
    "monument": "Monument status",
}


def utrecht_prediction_drivers(estimator, features, training_data, area_benchmark, limit=3):
    """Compare one input at a time with a typical training-data profile."""
    training = normalize_utrecht_data(training_data).copy()
    typical = {
        feature: float(training[feature].median())
        for feature in UTRECHT_NUMERIC_FEATURES
        if pd.notna(training[feature].median())
    }
    baseline = pd.DataFrame([typical], columns=UTRECHT_MODEL_FEATURES)
    estimate = float(estimator.predict(features)[0])
    drivers = []

    for feature in UTRECHT_MODEL_FEATURES:
        value = features.iloc[0][feature]
        baseline_value = baseline.iloc[0][feature]
        if pd.isna(value) or pd.isna(baseline_value):
            continue
        same_as_typical = math.isclose(float(value), float(baseline_value), abs_tol=1e-9)
        if same_as_typical:
            continue

        comparison = features.copy()
        comparison.at[comparison.index[0], feature] = baseline_value
        without_feature_difference = (
            estimate - float(estimator.predict(comparison)[0])
        ) * float(area_benchmark)
        if math.isclose(without_feature_difference, 0.0, abs_tol=500):
            continue

        if feature in {"lot-area", "house-area", "garden-size"}:
            display_value = f"{float(value):g} m²"
        elif feature in {"lot-len", "lot-width"}:
            display_value = f"{float(value):g} m"
        elif feature == "monument":
            display_value = "Yes" if float(value) == 1 else "No"
        else:
            display_value = f"{float(value):g}"

        drivers.append({
            "label": UTRECHT_FEATURE_LABELS[feature],
            "value": display_value,
            "difference": without_feature_difference,
        })

    return sorted(drivers, key=lambda item: abs(item["difference"]), reverse=True)[:limit]


def show_utrecht_prediction_reasons(estimator, features, training_data, area_benchmark):
    drivers = utrecht_prediction_drivers(
        estimator, features, training_data, area_benchmark
    )
    st.markdown("**What is moving the estimate?**")
    if drivers:
        columns = st.columns(len(drivers))
        for column, driver in zip(columns, drivers):
            direction = "higher" if driver["difference"] > 0 else "lower"
            with column:
                st.markdown(f"**{driver['label']}**")
                st.write(driver["value"])
                st.caption(
                    f"About €{abs(driver['difference']):,.0f} {direction} than a typical sample home"
                )
    else:
        st.caption("The available inputs are close to the sample’s typical property profile.")
    st.caption(
        "Each amount compares one property fact with a typical training profile while holding other inputs fixed. "
        "These are model sensitivities, not causes, and they do not need to add up to the total estimate."
    )


@st.dialog("Home listing", width="large", icon=":material/home:")
def show_listing_details(listing):
    photo_url = listing_photo_source(listing.get("photo_url"))
    if photo_url:
        st.image(photo_url, width="stretch")

    st.subheader(str(listing.get("title") or "Dutch home"))
    price = listing.get("price")
    price_label = "Monthly rent" if listing.get("offer_type") == "rent" else "Asking price"
    if pd.notna(price):
        price_text = f"€{float(price):,.0f}" + (" / month" if listing.get("offer_type") == "rent" else "")
        st.metric(price_label, price_text)

    detail_pairs = [
        ("Location", listing.get("city")),
        ("Property type", listing.get("property_type")),
        ("Living area", f"{float(listing['area_m2']):g} m²" if pd.notna(listing.get("area_m2")) else None),
        ("Plot area", f"{float(listing['lot_area_m2']):g} m²" if pd.notna(listing.get("lot_area_m2")) else None),
        ("Rooms", f"{float(listing['rooms']):g}" if pd.notna(listing.get("rooms")) else None),
        ("Bedrooms", f"{float(listing['bedrooms']):g}" if pd.notna(listing.get("bedrooms")) else None),
        ("Bathrooms", f"{float(listing['bathrooms']):g}" if pd.notna(listing.get("bathrooms")) else None),
        ("Build year", listing.get("build_year")),
        ("Garden", f"{float(listing['garden_area_m2']):g} m²" if pd.notna(listing.get("garden_area_m2")) else None),
        ("Solar panels", listing.get("solar_panels")),
        ("Energy label", listing.get("energy_label")),
    ]
    for label, value in detail_pairs:
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            st.write(f"**{label}:** {value}")

    description = listing.get("description")
    if isinstance(description, str) and description.strip():
        st.markdown("**Description**")
        st.write(description)

    source_url = listing.get("listing_url")
    if isinstance(source_url, str) and source_url.startswith(("https://", "http://")):
        st.link_button("Open source listing", source_url, icon=":material/open_in_new:")


def read_klavida_cache():
    if not KLAVIDA_CACHE_FILE.exists():
        return None
    try:
        return json.loads(KLAVIDA_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(ttl=300, max_entries=30, show_spinner=False)
def search_web(query, api_key):
    """Search the public web with Tavily and return results with their source URLs."""
    request_body = json.dumps({
        "query": query,
        "search_depth": "basic",
        "max_results": 6,
        "topic": "general",
        "include_answer": True,
        "include_raw_content": False,
        "include_images": False,
        "country": "netherlands",
    }).encode("utf-8")
    request = Request(
        TAVILY_SEARCH_URL,
        data=request_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=25) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Search service returned HTTP {error.code}: {detail[:500]}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach the web search service: {error.reason}") from error


def pdok_json(url):
    request = Request(url, headers={"User-Agent": "HouseLens local property explorer"})
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


@st.cache_data(ttl=3600, max_entries=100)
def search_dutch_addresses(query):
    params = urlencode({"q": query, "fq": "type:adres", "rows": 8})
    payload = pdok_json(f"{PDOK_LOCATIONS_URL}?{params}")
    return [
        item for item in payload.get("response", {}).get("docs", [])
        if item.get("type") == "adres"
    ]


@st.cache_data(ttl=86400, max_entries=200)
def load_bag_dwelling(dwelling_id):
    params = urlencode({"f": "json", "identificatie": dwelling_id, "limit": 1})
    payload = pdok_json(
        f"{BAG_API_URL}/collections/verblijfsobject/items?{params}"
    )
    features = payload.get("features", [])
    return features[0] if features else None


@st.cache_data(ttl=86400, max_entries=200)
def load_bag_building(pand_href):
    parsed = urlparse(pand_href)
    if parsed.hostname != "api.pdok.nl":
        return None
    feature_id = parsed.path.rstrip("/").split("/")[-1]
    url = f"{BAG_API_URL}/collections/pand/items/{quote(feature_id, safe='')}?f=json"
    return pdok_json(url)


@st.cache_data(ttl=86400, max_entries=200)
def load_cadastral_parcel(parcel_reference):
    parts = str(parcel_reference).split("-")
    if len(parts) != 3 or not parts[2].isdigit():
        return None
    municipality_code, section, parcel_number = parts
    params = urlencode({
        "f": "json",
        "limit": 1,
        "akr_kadastrale_gemeente_code_waarde": municipality_code,
        "sectie": section,
        "perceelnummer": int(parcel_number),
    })
    payload = pdok_json(f"{BRK_API_URL}/collections/perceel/items?{params}")
    features = payload.get("features", [])
    return features[0] if features else None


@st.cache_data(ttl=86400, max_entries=200)
def load_3dbag(building_identification):
    if not building_identification:
        return None
    feature_id = f"NL.IMBAG.Pand.{building_identification}"
    payload = pdok_json(f"{THREEDBAG_API_URL}/{quote(feature_id, safe='.')}")
    objects = payload.get("feature", {}).get("CityObjects", {})
    for city_object in objects.values():
        if city_object.get("type") == "Building":
            return city_object.get("attributes", {})
    return None


def klavida_items(payload):
    """Unwrap common JSON-LD list shapes used by the Klavida open feed."""
    if isinstance(payload, list):
        return [item.get("item", item) if isinstance(item, dict) else item for item in payload]
    if not isinstance(payload, dict):
        return []
    for key in ("@graph", "itemListElement", "items", "listings", "properties", "aanbod"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item.get("item", item) if isinstance(item, dict) else item for item in value]
    kind = payload.get("@type")
    is_listing = kind in {"RealEstateListing", "Offer"} if isinstance(kind, str) else (
        isinstance(kind, list) and any(value in {"RealEstateListing", "Offer"} for value in kind)
    )
    if is_listing:
        return [payload]
    return []


def klavida_listings(payload):
    """Map Klavida's reusable JSON-LD facts into the HouseLens listing schema."""
    records = []
    for listing in klavida_items(payload):
        if not isinstance(listing, dict):
            continue
        home = listing.get("itemOffered") or listing.get("mainEntity") or listing.get("about") or {}
        if not isinstance(home, dict):
            home = {}
        address = listing.get("address") or home.get("address") or {}
        if not isinstance(address, dict):
            address = {}
        offer = listing.get("offers") or home.get("offers") or {}
        if isinstance(offer, list):
            offer = offer[0] if offer else {}
        if not isinstance(offer, dict):
            offer = {}
        area = home.get("floorSize") or listing.get("floorSize") or home.get("livingArea") or {}
        if isinstance(area, dict):
            area = area.get("value")
        price = offer.get("price") or listing.get("price") or home.get("price")
        rooms = home.get("numberOfRooms") or listing.get("numberOfRooms")
        records.append({
            "offer_type": "buy",
            "title": listing.get("name") or home.get("name") or "Home for sale",
            "city": address.get("addressLocality") or address.get("addressRegion") or listing.get("city") or "",
            "price": price,
            "area_m2": area,
            "rooms": rooms,
            "photo_url": "",
            "listing_url": listing.get("url") or listing.get("mainEntityOfPage") or home.get("url") or "",
            "property_type": home.get("@type") or listing.get("propertyType") or "Home",
            "energy_label": listing.get("energyLabel") or home.get("energyLabel") or "",
            "source": "Klavida",
            "status": "available",
        })
    return pd.DataFrame(records, columns=LISTING_COLUMNS)


def refresh_klavida_feed():
    """Fetch the open feed only when the source's hourly refresh interval allows it."""
    previous = read_klavida_cache() or {}
    last_fetch = previous.get("fetched_at")
    if last_fetch:
        last_fetch_time = datetime.fromisoformat(last_fetch)
        next_refresh = last_fetch_time + KLAVIDA_REFRESH_INTERVAL
        if datetime.now(timezone.utc) < next_refresh:
            return False, next_refresh, "Klavida allows one feed refresh per hour."

    fetched_at = datetime.now(timezone.utc)
    try:
        request = Request(KLAVIDA_FEED_URL, headers={"User-Agent": "HouseLens local property demo"})
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        KLAVIDA_CACHE_FILE.write_text(
            json.dumps({"fetched_at": fetched_at.isoformat(), "payload": payload}, ensure_ascii=False),
            encoding="utf-8",
        )
        return True, fetched_at + KLAVIDA_REFRESH_INTERVAL, None
    except Exception as error:
        KLAVIDA_CACHE_FILE.write_text(
            json.dumps({"fetched_at": fetched_at.isoformat(), "payload": previous.get("payload"), "error": str(error)}, ensure_ascii=False),
            encoding="utf-8",
        )
        return False, fetched_at + KLAVIDA_REFRESH_INTERVAL, f"Could not load the Klavida feed: {error}"


def normalize_listings(source):
    """Convert a permitted listings feed or CSV to the HouseLens card fields."""
    if source is None or source.empty:
        return pd.DataFrame(columns=LISTING_COLUMNS)

    frame = source.copy()
    aliases = {
        "offer_type": ["transaction_type", "aanbod", "listing_type"],
        "title": ["name", "headline", "titel"],
        "city": ["plaats", "woonplaats", "municipality"],
        "price": ["asking_price", "monthly_rent", "prijs", "rent"],
        "area_m2": ["surface_m2", "living_area_m2", "area", "woonoppervlakte"],
        "lot_area_m2": ["plot_area_m2", "lot_area", "perceeloppervlakte"],
        "rooms": ["room_count", "number_of_rooms", "kamers"],
        "bedrooms": ["bedroom_count", "slaapkamers"],
        "bathrooms": ["bathroom_count", "badkamers"],
        "build_year": ["buildyear", "bouwjaar"],
        "garden_area_m2": ["garden_size_m2", "garden_area", "tuinoppervlakte"],
        "solar_panels": ["panel_count", "zonnepanelen"],
        "photo_url": ["image_url", "photo", "image", "foto", "fotos"],
        "listing_url": ["url", "source_url", "link"],
        "property_type": ["type", "woningtype"],
        "energy_label": ["energielabel"],
        "source": ["provider", "bron", "makelaar"],
        "status": ["availability", "beschikbaarheid"],
    }
    existing = {str(column).strip().lower(): column for column in frame.columns}
    rename = {}
    for target, names in aliases.items():
        if target in frame.columns:
            continue
        for name in [target, *names]:
            if name.lower() in existing:
                rename[existing[name.lower()]] = target
                break
    frame = frame.rename(columns=rename)

    for column, default in {
        "offer_type": "buy",
        "title": "Dutch home",
        "city": "",
        "price": 0,
        "area_m2": None,
        "rooms": None,
        "photo_url": "",
        "listing_url": "",
        "property_type": "Home",
        "energy_label": "",
        "source": "Listing provider",
        "status": "available",
        "description": "",
        "lot_area_m2": None,
        "bedrooms": None,
        "bathrooms": None,
        "build_year": None,
        "garden_area_m2": None,
        "solar_panels": None,
    }.items():
        if column not in frame.columns:
            frame[column] = default

    frame["title"] = frame["title"].fillna("Dutch home")
    frame["city"] = frame["city"].fillna("")
    frame["property_type"] = frame["property_type"].fillna("Home")
    frame["source"] = frame["source"].fillna("Listing provider")

    def parse_number(value):
        if pd.isna(value):
            return None
        text = re.sub(r"[^0-9,.]", "", str(value))
        if not text:
            return None
        if "," in text and "." in text:
            if text.rfind(",") > text.rfind("."):
                text = text.replace(".", "").replace(",", ".")
            else:
                text = text.replace(",", "")
        elif "," in text:
            decimals = len(text.rsplit(",", 1)[1])
            text = text.replace(",", ".") if decimals == 2 else text.replace(",", "")
        elif "." in text and len(text.rsplit(".", 1)[1]) == 3:
            text = text.replace(".", "")
        try:
            return float(text)
        except ValueError:
            return None

    offer = frame["offer_type"].astype(str).str.strip().str.lower()
    frame["offer_type"] = offer.replace({
        "koop": "buy", "te koop": "buy", "koopwoning": "buy", "for sale": "buy", "sale": "buy",
        "huur": "rent", "huurwoning": "rent", "for rent": "rent", "rental": "rent",
    })
    frame["price"] = frame["price"].map(parse_number)
    for column in [
        "area_m2", "lot_area_m2", "rooms", "bedrooms", "bathrooms",
        "build_year", "garden_area_m2", "solar_panels",
    ]:
        frame[column] = frame[column].map(parse_number)
    frame["status"] = frame["status"].astype(str).str.strip().str.lower()
    frame["status"] = frame["status"].replace({
        "verkocht onder voorbehoud": "under offer",
        "onder bod": "under offer",
        "sold subject to contract": "under offer",
    })
    inactive = {"sold", "rented", "let", "inactive", "removed", "verkocht", "verhuurd", "ingetrokken"}
    frame = frame[~frame["status"].isin(inactive)]
    return frame[frame["offer_type"].isin(["buy", "rent"])].copy()


if not st.session_state.get("show_houselens_app", False):
    st.markdown(
        """
        <style>
          .block-container {max-width: 1180px; padding-top: 1.2rem;}
          .hl-nav {display:flex; align-items:center; justify-content:space-between; margin:0 auto 5rem; max-width:980px;}
          .hl-brand {color:#347cf6; font-size:1.35rem; font-weight:750; letter-spacing:-.03em;}
          .hl-nav-note {color:#747986; font-size:.95rem;}
          .hl-hero {text-align:center; margin:0 auto; max-width:800px;}
          .hl-eyebrow {color:#7b8494; font-size:.82rem; letter-spacing:.09em; text-transform:uppercase;}
          .hl-title {color:#1d2028; font-size:clamp(2.7rem, 5vw, 4.15rem); line-height:1.05; letter-spacing:-.055em; font-weight:750; margin:.9rem 0 1rem;}
          .hl-subtitle {color:#687080; font-size:1.2rem; line-height:1.55; margin:0 auto 1.4rem; max-width:650px;}
          .hl-trust {color:#8991a0; font-size:.88rem; text-align:center; margin-top:.75rem;}
          .hl-stats {display:flex; justify-content:center; gap:5rem; border-top:1px solid #e8eaf0; margin:3.4rem auto 2rem; padding:2rem 1rem 0; max-width:900px; text-align:center;}
          .hl-stat-title {color:#242733; font-size:1.05rem; font-weight:700;}
          .hl-stat-copy {color:#858b98; font-size:.85rem; margin-top:.3rem;}
          @media(max-width:700px) {.hl-nav{margin-bottom:3rem}.hl-stats{gap:1rem; flex-wrap:wrap}}
        </style>
        <div class="hl-nav">
          <div class="hl-brand">⌂ HouseLens</div>
          <div class="hl-nav-note">Dutch housing · Market insights · Home estimates</div>
        </div>
        <div class="hl-hero">
          <div class="hl-eyebrow">Dutch housing market intelligence</div>
          <div class="hl-title">A clearer picture of your next home.</div>
          <div class="hl-subtitle">Explore Dutch homes, understand the local market, and see which property details shape an estimate.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left_space, form_center, right_space = st.columns([1, 7, 1])
    with form_center:
        with st.form("houselens_home_search"):
            home_goal = st.segmented_control(
                "I'm looking to",
                options=["Buy", "Rent"],
                default="Buy",
                key="home_housing_goal",
                width="content",
            )
            address_col, button_col = st.columns([5, 1.25], vertical_alignment="bottom")
            with address_col:
                home_address = st.text_input(
                    "Property address",
                    placeholder="Enter a Dutch address or town",
                    label_visibility="collapsed",
                    key="homepage_address_input",
                )
            with button_col:
                check_home = st.form_submit_button("Check value", type="primary", width="stretch")
    if check_home:
        if not home_address.strip():
            st.warning("Enter an address or town to continue.")
        else:
            st.session_state["housing_goal"] = home_goal
            st.session_state["homepage_search_pending"] = home_address.strip()
            st.session_state["show_houselens_app"] = True
            st.rerun()
    st.markdown(
        '<div class="hl-trust">⌖ Official property facts &nbsp;&nbsp; · &nbsp;&nbsp; ↗ Local price trends &nbsp;&nbsp; · &nbsp;&nbsp; ◇ No sign-up</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hl-stats">
          <div><div class="hl-stat-title">Official Dutch records</div><div class="hl-stat-copy">Address and building details</div></div>
          <div><div class="hl-stat-title">Local market context</div><div class="hl-stat-copy">CBS municipality price trends</div></div>
          <div><div class="hl-stat-title">Buy or rent</div><div class="hl-stat-copy">Explore the housing view</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

pending_home_search = st.session_state.pop("homepage_search_pending", None)
if pending_home_search:
    st.session_state["address_search_query"] = pending_home_search

st.title("Find your place in the Netherlands", icon=":material/home:")
st.caption("Browse homes to buy or rent, then compare the details that matter to you.")

housing_goal = st.segmented_control(
    "I'm looking to",
    options=["Buy", "Rent"],
    default="Buy",
    key="housing_goal",
    width="content",
)

if not MODEL_FILE.exists():
    st.error("The trained model was not found. Run `python train_model.py` first.")
    st.stop()

info, trend, data = load_files()
model = load_model()
utrecht_training_data = load_utrecht_training_data(
    (UTRECHT_TRAINING_FILE.stat().st_mtime_ns, UTRECHT_TRAINING_FILE.stat().st_size)
    if UTRECHT_TRAINING_FILE.exists()
    else None
)
sale_calibrations = load_sale_calibrations()
try:
    cbs_municipality_prices = load_cbs_municipality_prices()
except Exception:
    cbs_municipality_prices = pd.DataFrame(
        columns=[
            "municipality_code", "municipality", "average_price_eur", "year",
            "market_adjustment_factor", "market_adjustment_pct", "market_index_period",
            "market_adjusted_price_eur",
        ]
    )

with st.sidebar:
    if st.button("HouseLens home", icon=":material/home:", width="stretch"):
        st.session_state["show_houselens_app"] = False
        st.rerun()
    st.header("Ames demo model · Iowa")
    st.write(f"Selected after testing: **{info['selected_model']}**")
    st.write(f"Test-set MAE: **${info['selected_model_test_mae_usd']:,.0f}**")
    st.caption("Ames results are in U.S. dollars. Dutch estimates use a CBS municipality price baseline with a small Utrecht/Amersfoort property-feature model.")

tab_browse, tab_predict, tab_trends, tab_results, tab_search = st.tabs([
    "Browse homes", "Price estimate demo", "Past prices demo", "Model results demo", "Web search"
])

with tab_browse:
    st.subheader("Housing data" if housing_goal == "Buy" else "Rental homes")
    st.caption(
        "Search Dutch building records and browse permitted listings. Sale estimates use a CBS municipality benchmark. "
        "The sale feed checks automatically "
        "no more than once per hour; it does not provide photos, and a rental feed is not connected."
    )
    if st.button(
        "Refresh CBS market data",
        key="refresh_cbs_market_data",
        icon=":material/refresh:",
    ):
        load_cbs_municipality_index_history.clear()
        load_cbs_municipality_index_adjustments.clear()
        load_cbs_municipality_prices.clear()
        st.rerun()

    with st.form("dutch_address_search"):
        address_query = st.text_input(
            "Search Dutch addresses or streets",
            placeholder="For example: Merelhoven 343, Capelle aan den IJssel",
            help="Looks up registered addresses nationwide and shows official BAG, cadastral, and 3D building facts.",
            key="address_search_query",
        )
        search_addresses = st.form_submit_button(
            "Search Dutch housing records",
            type="primary",
            icon=":material/search:",
        ) or bool(pending_home_search)

    if search_addresses:
        if not address_query.strip():
            st.warning("Enter an address, street, postcode, or town to search.")
        else:
            try:
                with st.spinner("Searching Dutch public property records…"):
                    matches = search_dutch_addresses(address_query.strip())
                st.session_state["bag_address_matches"] = matches
                st.session_state["bag_address_query"] = address_query.strip()
                st.session_state.pop("bag_address_error", None)
                if not matches:
                    st.info("No registered address matches that search. Try adding the postcode or town.")
            except Exception as error:
                st.session_state["bag_address_matches"] = []
                st.session_state["bag_address_error"] = str(error)

    if st.session_state.get("bag_address_error"):
        st.error(f"Could not reach the Dutch property-data services: {st.session_state['bag_address_error']}")

    address_matches = st.session_state.get("bag_address_matches", [])
    if address_matches:
        st.markdown("#### Official property record")
        st.caption(
            f"Choose one of up to {len(address_matches)} matching registered addresses. "
            "These are building records, not proof that the home is currently for sale or rent."
        )
        selected_address_index = st.selectbox(
            "Matching registered addresses",
            options=list(range(len(address_matches))),
            format_func=lambda index: address_matches[index].get("weergavenaam", "Registered address"),
            key="bag_address_choice",
        )
        selected_address = address_matches[selected_address_index]
        dwelling = None
        bag_building = None
        parcel = None
        three_d_facts = None
        try:
            with st.spinner("Loading BAG, cadastral, and 3D building details…"):
                dwelling_id = selected_address.get("adresseerbaarobject_id")
                if dwelling_id:
                    dwelling = load_bag_dwelling(dwelling_id)
                if dwelling:
                    bag_props = dwelling.get("properties", {})
                    pand_hrefs = bag_props.get("pand.href") or []
                    if isinstance(pand_hrefs, str):
                        pand_hrefs = [pand_hrefs]
                    if pand_hrefs:
                        bag_building = load_bag_building(pand_hrefs[0])
                    parcel_refs = selected_address.get("gekoppeld_perceel") or []
                    if isinstance(parcel_refs, str):
                        parcel_refs = [parcel_refs]
                    if parcel_refs:
                        parcel = load_cadastral_parcel(parcel_refs[0])
                    building_id = (bag_building or {}).get("properties", {}).get("identificatie")
                    if building_id:
                        three_d_facts = load_3dbag(building_id)
        except Exception as error:
            st.warning(f"Some public registry details could not be loaded: {error}")

        with st.container(border=True):
            st.badge("Official BAG address", color="blue")
            st.subheader(selected_address.get("weergavenaam", "Registered Dutch address"))
            if dwelling:
                bag_props = dwelling.get("properties", {})
                building_props = (bag_building or {}).get("properties", {})
                parcel_props = (parcel or {}).get("properties", {})
                official_area = bag_props.get("oppervlakte")
                build_year = building_props.get("bouwjaar")
                plot_area = parcel_props.get("kadastrale_grootte_waarde")
                area_cols = st.columns(3)
                with area_cols[0]:
                    st.metric("BAG floor area", f"{official_area} m²" if official_area else "Unavailable")
                with area_cols[1]:
                    st.metric("Build year", str(build_year) if build_year else "Unavailable")
                with area_cols[2]:
                    st.metric("Cadastral parcel", f"{plot_area} m²" if plot_area else "Unavailable")
                st.caption(
                    f"Use: {bag_props.get('gebruiksdoel', 'not recorded')} · "
                    f"BAG status: {bag_props.get('status', 'not recorded')}"
                )
                if housing_goal == "Buy":
                    if utrecht_training_data.empty:
                        st.warning("The Dutch price model could not load its local training data.")
                    elif official_area is None:
                        st.info("A BAG floor area is needed to estimate a sale value for this address.")
                    else:
                        area_price = find_cbs_area_price(
                            cbs_municipality_prices,
                            selected_address.get("gemeentecode"),
                            selected_address.get("gemeentenaam"),
                        )
                        if area_price is None:
                            st.info(
                                "A CBS sale-price benchmark could not be matched to this municipality, "
                                "so HouseLens is not showing an unsupported address estimate."
                            )
                        else:
                            feature_factor, estimate_features, estimator = predict_address_value(
                                dwelling, bag_building, parcel, utrecht_training_data
                            )
                            area_average = float(area_price["average_price_eur"])
                            market_baseline = float(area_price["market_adjusted_price_eur"])
                            market_change = area_price["market_adjustment_pct"]
                            market_period = area_price["market_index_period"]
                            model_prediction = market_baseline * feature_factor
                            sale_reference = find_sale_calibration(
                                selected_address.get("weergavenaam"), sale_calibrations
                            )
                            market_note = (
                                f" through {market_period}."
                                if isinstance(market_period, str)
                                else ". No local index update was available."
                            )
                            with st.container(border=True):
                                title = (
                                    "**HouseLens sale-informed value**"
                                    if sale_reference is not None
                                    else "**HouseLens estimated sale value**"
                                )
                                st.markdown(title)
                                estimate_cols = st.columns([1.0, 1.0, 0.8, 1.45])
                                with estimate_cols[0]:
                                    if sale_reference is not None:
                                        sale_floor = float(sale_reference["sale_price_lower_bound_eur"])
                                        st.metric("Sale-informed value floor", f"€{sale_floor:,.0f}+")
                                    else:
                                        st.metric("HouseLens estimate", f"€{model_prediction:,.0f}")
                                with estimate_cols[1]:
                                    st.metric("CBS baseline after index", f"€{market_baseline:,.0f}")
                                with estimate_cols[2]:
                                    st.metric(
                                        "Market update",
                                        f"{market_change:+.1f}%" if pd.notna(market_change) else "Unavailable",
                                        delta=market_period if isinstance(market_period, str) else None,
                                    )
                                with estimate_cols[3]:
                                    show_cbs_municipality_price_trend(area_price["municipality_code"])
                                municipality = str(area_price["municipality"]).replace(" (gemeente)", "")
                                st.caption(
                                    f"CBS area baseline for {municipality}: €{market_baseline:,.0f}, starting from the "
                                    f"{int(area_price['year'])} average of €{area_average:,.0f} and updated by the local house-price index"
                                    f"{market_note} The feature adjustment comes from a Random Forest trained on {len(utrecht_training_data):,} "
                                    f"sample records from Utrecht and Amersfoort postcode areas ({feature_factor:.2f}×). "
                                    "It learns property-feature differences within those areas and does not use tax values."
                                )
                                st.caption(
                                    "The CBS figure is an average of existing homes, not a comparable sale for this address. "
                                    "This demonstration does not include a Capelle-specific property-sales training set, "
                                    "so this is not a formal valuation. Missing house details use sample medians."
                                )
                                if sale_reference is not None:
                                    st.info(
                                        f"This address has a user-reported sale price above €{sale_floor:,.0f}. "
                                        f"The feature model alone estimates €{model_prediction:,.0f}; the sale figure is shown "
                                        "as a property-specific lower-bound reference, not as an independently predicted exact price. "
                                        "It does not change estimates for other homes."
                                    )
                                show_utrecht_prediction_reasons(
                                    estimator, estimate_features, utrecht_training_data, market_baseline
                                )
                                st.link_button(
                                    "CBS price data source",
                                    CBS_MUNICIPAL_PRICE_PAGE,
                                    icon=":material/open_in_new:",
                                )
                                st.link_button(
                                    "CBS municipality price index",
                                    CBS_MUNICIPAL_INDEX_PAGE,
                                    icon=":material/open_in_new:",
                                )
                if three_d_facts:
                    st.markdown("**3DBAG building model**")
                    model_cols = st.columns(3)
                    with model_cols[0]:
                        floors = three_d_facts.get("b3_bouwlagen")
                        st.metric("Modelled floors", str(floors) if floors is not None else "Unavailable")
                    with model_cols[1]:
                        volume = three_d_facts.get("b3_volume_lod13")
                        st.metric("Modelled volume", f"{volume:,.0f} m³" if volume else "Unavailable")
                    with model_cols[2]:
                        footprint = three_d_facts.get("b3_opp_grond")
                        st.metric("Building footprint", f"{footprint:g} m²" if footprint else "Unavailable")
                point = dwelling.get("geometry", {}).get("coordinates")
                if isinstance(point, list) and len(point) >= 2:
                    map_layers = []
                    if parcel and parcel.get("geometry"):
                        map_layers.append(pdk.Layer(
                            "GeoJsonLayer",
                            data={
                                "type": "Feature",
                                "geometry": parcel["geometry"],
                                "properties": parcel_props,
                            },
                            get_fill_color=[37, 99, 235, 35],
                            get_line_color=[37, 99, 235, 230],
                            line_width_min_pixels=2,
                            filled=True,
                            stroked=True,
                            pickable=False,
                        ))
                    map_layers.append(pdk.Layer(
                        "ScatterplotLayer",
                        data=[{"lon": point[0], "lat": point[1]}],
                        get_position="[lon, lat]",
                        get_color=[15, 118, 110, 220],
                        get_radius=6,
                        radius_min_pixels=6,
                        radius_max_pixels=10,
                    ))
                    st.pydeck_chart(
                        pdk.Deck(
                            map_style=None,
                            initial_view_state=pdk.ViewState(
                                latitude=point[1], longitude=point[0], zoom=18, pitch=0
                            ),
                            layers=map_layers,
                        ),
                        height=320,
                    )
                    if parcel:
                        st.caption("Map outline is from the public cadastral map and is for location context, not a boundary survey.")
                source_cols = st.columns(3)
                bag_url = (
                    f"{BAG_API_URL}/collections/verblijfsobject/items?f=html&"
                    f"{urlencode({'identificatie': dwelling_id})}"
                )
                with source_cols[0]:
                    st.link_button("Open BAG record", bag_url, icon=":material/open_in_new:")
                if parcel:
                    parcel_params = urlencode({
                        "akr_kadastrale_gemeente_code_waarde": parcel_props.get("akr_kadastrale_gemeente_code_waarde", ""),
                        "sectie": parcel_props.get("sectie", ""),
                        "perceelnummer": parcel_props.get("perceelnummer", ""),
                    })
                    with source_cols[1]:
                        st.link_button(
                            "Open cadastral data",
                            f"{BRK_API_URL}/collections/perceel/items?f=html&{parcel_params}",
                            icon=":material/open_in_new:",
                        )
                if bag_building:
                    building_id = bag_building.get("properties", {}).get("identificatie")
                    if building_id:
                        with source_cols[2]:
                            st.link_button(
                                "Open 3DBAG model",
                                f"{THREEDBAG_API_URL}/NL.IMBAG.Pand.{building_id}",
                                icon=":material/open_in_new:",
                            )
                st.caption(
                    "Official registries provide address, area, year, parcel, and modelled building facts. "
                    "They do not provide current asking prices, room counts, availability, or listing photos. "
                    "BAG floor area can differ from an advert's measured living area, so HouseLens keeps the sources separate."
                )
                st.caption("Sources: PDOK/Kadaster Locatieserver, BAG and BRK; 3DBAG building model (CC BY 4.0).")
            else:
                st.info("The address is registered, but its BAG dwelling details were not returned by the service.")

    klavida_cache = read_klavida_cache() or {}
    last_fetch = klavida_cache.get("fetched_at")
    if housing_goal == "Buy":
        refresh_due = True
        if last_fetch:
            try:
                refresh_due = datetime.now(timezone.utc) >= (
                    datetime.fromisoformat(last_fetch) + KLAVIDA_REFRESH_INTERVAL
                )
            except ValueError:
                refresh_due = True
        if refresh_due:
            with st.spinner("Checking the permitted sale-listing feed…"):
                refresh_klavida_feed()
            klavida_cache = read_klavida_cache() or klavida_cache
    klavida_payload = klavida_cache.get("payload")

    listings_version = (
        (LISTINGS_FILE.stat().st_mtime_ns, LISTINGS_FILE.stat().st_size)
        if LISTINGS_FILE.exists()
        else None
    )
    catalog_listings = load_listings_file(listings_version)
    feed_listings = (
        klavida_listings(klavida_payload)
        if housing_goal == "Buy" and klavida_payload is not None
        else pd.DataFrame(columns=LISTING_COLUMNS)
    )
    raw_listings = pd.concat([catalog_listings, feed_listings], ignore_index=True, sort=False)
    listings = normalize_listings(raw_listings)
    if not listings.empty:
        has_source_url = listings["listing_url"].fillna("").astype(str).str.strip().ne("")
        listings = pd.concat(
            [
                listings.loc[~has_source_url],
                listings.loc[has_source_url].drop_duplicates(subset=["listing_url"], keep="last"),
            ],
            ignore_index=True,
        )
    listings = listings[listings["offer_type"] == ("buy" if housing_goal == "Buy" else "rent")]

    if listings.empty:
        with st.container(border=True):
            if housing_goal == "Rent":
                st.subheader("Rental listings are not connected")
                st.write(
                    "The automatic provider feed currently supplies sale listings only. "
                    "The Dutch address and building facts still work, but there is no rental feed or rent model connected."
                )
            elif klavida_cache.get("error"):
                st.subheader("The automatic sale feed could not be refreshed")
                st.write(str(klavida_cache["error"]))
            else:
                st.subheader("No current sale listings in the connected feed")
                st.write(
                    "HouseLens checks Klavida automatically, at most once per hour. "
                    "Its latest open feed contains no listing records. You can still search official property records above."
                )
            st.caption(
                "The open feed does not include listing photos. Address registries do not provide asking prices, "
                "availability, room counts, or photographs."
            )

    if not listings.empty:
        filter_city, filter_price, filter_rooms = st.columns([1.3, 1, 1])
        with filter_city:
            cities = ["All locations", *sorted(listings["city"].fillna("").astype(str).unique())]
            selected_city = st.selectbox("Location", cities)
        with filter_price:
            max_price = int(listings["price"].dropna().max()) if listings["price"].notna().any() else 0
            price_limit = st.number_input(
                "Maximum price" + (" per month" if housing_goal == "Rent" else ""),
                min_value=0,
                max_value=max(max_price, 0),
                value=max(max_price, 0),
                step=100 if housing_goal == "Rent" else 10000,
            )
        with filter_rooms:
            minimum_rooms = st.selectbox("Minimum rooms", ["Any", 1, 2, 3, 4, 5])

        filtered = listings.copy()
        if address_query.strip():
            query = address_query.strip().casefold()
            searchable = (
                filtered[["title", "city", "listing_url"]]
                .fillna("")
                .astype(str)
                .agg(" ".join, axis=1)
                .str.casefold()
            )
            filtered = filtered[searchable.str.contains(query, regex=False)]
        if selected_city != "All locations":
            filtered = filtered[filtered["city"].astype(str) == selected_city]
        if price_limit:
            filtered = filtered[filtered["price"].fillna(0) <= price_limit]
        if minimum_rooms != "Any":
            filtered = filtered[filtered["rooms"].fillna(0) >= minimum_rooms]
        st.caption(f"{len(filtered)} homes found · listing photos and details appear when supplied by the source.")

        if filtered.empty:
            st.info("No homes match these filters. Try a higher budget or a different location.")
        else:
            cards = st.columns(3)
            for index, (_, listing) in enumerate(filtered.iterrows()):
                with cards[index % len(cards)]:
                    with st.container(border=True):
                        photo = listing_photo_source(listing.get("photo_url"))
                        if photo:
                            st.image(photo, width="stretch")
                        else:
                            st.info("Photo not supplied by listing source", icon=":material/image:")
                        listing_status = str(listing.get("status", "available")).strip().lower()
                        if listing_status == "under offer":
                            st.badge("Under offer", color="orange")
                        else:
                            st.badge("For sale" if housing_goal == "Buy" else "For rent", color="blue")
                        st.markdown(f"**{listing['title']}**")
                        price_text = f"€{listing['price']:,.0f}"
                        if housing_goal == "Rent":
                            price_text += " / month"
                        st.markdown(f"### {price_text}")
                        details = [str(listing["city"]), str(listing["property_type"])]
                        if pd.notna(listing["area_m2"]):
                            details.append(f"{listing['area_m2']:g} m²")
                        if pd.notna(listing.get("bedrooms")):
                            details.append(f"{listing['bedrooms']:g} bedrooms")
                        if pd.notna(listing["rooms"]):
                            details.append(f"{listing['rooms']:g} rooms")
                        if str(listing["energy_label"]).strip():
                            details.append(f"Label {listing['energy_label']}")
                        st.caption(" · ".join(value for value in details if value and value.lower() != "nan"))
                        sale_reference = find_sale_calibration(
                            f"{listing.get('title', '')}, {listing.get('city', '')}",
                            sale_calibrations,
                        )
                        if sale_reference is not None:
                            sale_floor = float(sale_reference["sale_price_lower_bound_eur"])
                            st.caption(
                                f"User-reported sale: €{sale_floor:,.0f}+ (lower bound; exact amount and date unknown)."
                            )
                        description = listing.get("description")
                        if isinstance(description, str) and description.strip():
                            st.write(description)
                        st.caption(f"Source: {listing['source']}")
                        listing_url = str(listing["listing_url"]).strip()
                        if listing_url and listing_url.lower() not in {"nan", "none"}:
                            st.link_button("View original listing", listing_url, width="stretch", icon=":material/open_in_new:")
                        if st.button(
                            "View home details",
                            key=f"listing_details_{index}",
                            width="stretch",
                            icon=":material/info:",
                        ):
                            show_listing_details(listing.to_dict())

with tab_predict:
    st.subheader("Dutch home estimate")
    if utrecht_training_data.empty:
        st.error("The local Utrecht training dataset could not be loaded.")
    elif cbs_municipality_prices.empty:
        st.error("The CBS municipality price benchmark is unavailable, so HouseLens cannot calculate a location-based Dutch estimate right now.")
    else:
        st.write(
            "Choose a municipality and enter the home’s details. HouseLens starts with that municipality’s latest CBS average sale price, then adjusts it using a model trained on property features. "
            "This model is separate from the Ames, Iowa demo below."
        )
        st.caption(
            f"Feature model: {len(utrecht_training_data):,} sample records from Utrecht and Amersfoort. "
            "Training targets are normalized within each sample postcode area so the model learns house-feature differences, "
            "not postcode prices. The CBS municipal average supplies the location baseline. Tax values are not used."
        )

        def median_or(column, fallback=0.0):
            value = utrecht_training_data[column].median()
            return float(value) if pd.notna(value) else float(fallback)

        energy_options = sorted(utrecht_training_data["energy-eff"].dropna().unique().tolist()) or [0, 1]
        monument_options = sorted(utrecht_training_data["monument"].dropna().unique().tolist()) or [0, 1]
        municipality_codes = cbs_municipality_prices.sort_values("municipality")["municipality_code"].tolist()
        municipality_names = dict(zip(
            cbs_municipality_prices["municipality_code"],
            cbs_municipality_prices["municipality"].astype(str).str.replace(" (gemeente)", "", regex=False),
        ))
        with st.form("utrecht_value_estimate"):
            location_col, size_col, condition_col = st.columns(3)
            with location_col:
                estimate_municipality = st.selectbox(
                    "Municipality",
                    municipality_codes,
                    format_func=lambda code: municipality_names.get(code, code),
                )
                estimate_buildyear = st.number_input(
                    "Build year", min_value=1800, max_value=2026,
                    value=int(median_or("buildyear", 1969)), step=1,
                )
                estimate_bathrooms = st.number_input(
                    "Bathrooms", min_value=0, max_value=10,
                    value=int(median_or("bathrooms")), step=1,
                )
                estimate_balcony = st.number_input(
                    "Balcony count", min_value=0, max_value=10,
                    value=int(median_or("balcony")), step=1,
                )
            with size_col:
                estimate_house_area = st.number_input(
                    "House area (m²)", min_value=1.0,
                    value=max(1.0, median_or("house-area", 100)), step=5.0,
                )
                estimate_lot_area = st.number_input(
                    "Lot area (m²)", min_value=0.0,
                    value=max(0.0, median_or("lot-area")), step=5.0,
                )
                estimate_garden = st.number_input(
                    "Garden size (m²)", min_value=0.0,
                    value=max(0.0, median_or("garden-size")), step=5.0,
                )
            with condition_col:
                estimate_lot_length = st.number_input(
                    "Lot length (m)", min_value=0.0,
                    value=max(0.0, median_or("lot-len")), step=0.5,
                )
                estimate_lot_width = st.number_input(
                    "Lot width (m)", min_value=0.0,
                    value=max(0.0, median_or("lot-width")), step=0.5,
                )
                estimate_energy = st.selectbox("Energy-efficiency source code", energy_options)
                estimate_monument = st.selectbox(
                    "Monument flag", monument_options,
                    format_func=lambda value: "Yes" if value == 1 else "No",
                )
            estimate_utrecht = st.form_submit_button("Estimate Dutch home value", type="primary")

        if estimate_utrecht:
            area_price = cbs_municipality_prices[
                cbs_municipality_prices["municipality_code"] == estimate_municipality
            ].iloc[0]
            area_average = float(area_price["average_price_eur"])
            market_baseline = float(area_price["market_adjusted_price_eur"])
            market_change = area_price["market_adjustment_pct"]
            market_period = area_price["market_index_period"]
            utrecht_model, training_rows = train_utrecht_model(utrecht_training_data)
            example_home = pd.DataFrame(
                [{
                    "lot-len": estimate_lot_length,
                    "lot-width": estimate_lot_width,
                    "lot-area": estimate_lot_area,
                    "house-area": estimate_house_area,
                    "garden-size": estimate_garden,
                    "balcony": estimate_balcony,
                    "buildyear": estimate_buildyear,
                    "bathrooms": estimate_bathrooms,
                    "energy-eff": estimate_energy,
                    "monument": estimate_monument,
                }],
                columns=UTRECHT_MODEL_FEATURES,
            )
            feature_factor = float(utrecht_model.predict(example_home)[0])
            prediction = market_baseline * feature_factor
            estimate_cols = st.columns([1.0, 1.0, 0.8, 1.45])
            with estimate_cols[0]:
                st.metric("HouseLens estimate", f"€{prediction:,.0f}")
            with estimate_cols[1]:
                st.metric("CBS baseline after index", f"€{market_baseline:,.0f}")
            with estimate_cols[2]:
                st.metric(
                    "Market update",
                    f"{market_change:+.1f}%" if pd.notna(market_change) else "Unavailable",
                    delta=market_period if isinstance(market_period, str) else None,
                )
            with estimate_cols[3]:
                show_cbs_municipality_price_trend(estimate_municipality)
            municipality = str(area_price["municipality"]).replace(" (gemeente)", "")
            market_note = (
                f" through {market_period}."
                if isinstance(market_period, str)
                else ". No local index update was available."
            )
            st.caption(
                f"CBS area baseline for {municipality}: €{market_baseline:,.0f}, starting from the "
                f"{int(area_price['year'])} average of €{area_average:,.0f} and updated by the local price index{market_note} "
                f"Feature model trained on {training_rows} Utrecht/Amersfoort sample records; feature adjustment {feature_factor:.2f}×. "
                "Missing facts on address lookups are filled with sample medians. This is a demonstration, not an appraisal."
            )
            show_utrecht_prediction_reasons(
                utrecht_model, example_home, utrecht_training_data, market_baseline
            )
            st.link_button(
                "CBS price data source",
                CBS_MUNICIPAL_PRICE_PAGE,
                icon=":material/open_in_new:",
            )
            st.link_button(
                "CBS municipality price index",
                CBS_MUNICIPAL_INDEX_PAGE,
                icon=":material/open_in_new:",
            )

    st.divider()
    st.subheader("Ames, Iowa estimate demo")
    st.write("Change the values to see how the historical U.S. model responds.")
    left, right = st.columns(2)
    with left:
        quality = st.slider("Overall quality (1 = poor, 10 = excellent)", 1, 10, 6)
        living_area = st.number_input("Above-ground living area (square feet)", 500, 5000, 1600, step=50)
        garage_cars = st.selectbox("Garage capacity (cars)", [0, 1, 2, 3, 4], index=2)
        basement_area = st.number_input("Basement area (square feet)", 0, 3000, 800, step=50)
    with right:
        full_baths = st.selectbox("Full bathrooms", [0, 1, 2, 3, 4], index=2)
        year_built = st.slider("Year built", 1872, 2010, 2000)
        lot_area = st.number_input("Lot size (square feet)", 1300, 200000, 8000, step=500)
        neighborhoods = sorted(data["Neighborhood"].dropna().unique().tolist())
        neighborhood = st.selectbox("Ames neighborhood", neighborhoods, index=neighborhoods.index("NAmes") if "NAmes" in neighborhoods else 0)

    if st.button("Estimate Ames price", type="secondary"):
        house = pd.DataFrame([{
            "OverallQual": quality,
            "GrLivArea": living_area,
            "GarageCars": garage_cars,
            "TotalBsmtSF": basement_area,
            "FullBath": full_baths,
            "YearBuilt": year_built,
            "LotArea": lot_area,
            "Neighborhood": neighborhood,
        }])
        estimate = float(model.predict(house)[0])
        margin = info["selected_model_test_mae_usd"]
        st.metric("Estimated historical sale price", f"${estimate:,.0f}")
        st.write(f"A rough reference using the test-set average error is **${max(0, estimate-margin):,.0f} to ${estimate+margin:,.0f}**. This is not a formal prediction interval.")
        st.caption("This model learned from past Ames sales. It has not been trained on Dutch homes.")

with tab_trends:
    st.subheader("Historical median sale prices")
    selected_area = st.selectbox("Choose an Ames neighborhood", sorted(trend["Neighborhood"].unique()))
    area_data = trend[trend["Neighborhood"] == selected_area].sort_values("Year")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(area_data["Year"], area_data["MedianSalePrice"], marker="o", color="#2563eb")
    ax.set_xlabel("Year sold")
    ax.set_ylabel("Median sale price (USD)")
    ax.set_title(f"Past sales in {selected_area}, Ames")
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    st.caption("The dataset covers sales from 2006 to 2010. Some neighborhoods have few sales in a year, so the median can move a lot. This chart is historical context, not a forecast.")

with tab_results:
    st.subheader("Model comparison on held-out houses")
    results = pd.read_csv(ROOT / "reports" / "model_results.csv")
    st.dataframe(results, hide_index=True, width="stretch")
    st.caption("The test set was set aside before training. MAE and RMSE are in US dollars; lower is better. R2 is a fit score; closer to 1 is better. The very simple median model is included as a baseline.")
    st.subheader("What the selected model paid attention to")
    rf = model.named_steps["model"]
    transformed = model.named_steps["prepare"]
    cat_names = transformed.named_transformers_["neighborhood"].named_steps["one_hot"].get_feature_names_out(["Neighborhood"]).tolist()
    feature_names = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt", "LotArea"] + cat_names
    importance = pd.Series(rf.feature_importances_, index=feature_names)
    grouped = importance.groupby([name.split("_")[0] for name in importance.index]).sum().sort_values()
    st.bar_chart(grouped)
    st.caption("Feature importance describes how the random forest used the training features. It does not prove that a feature causes a higher price.")

with tab_search:
    st.subheader("Search the web")
    st.caption(
        "Searches public web pages and shows source-linked summaries. Search results do not automatically "
        "copy listing photos or import a home into your listings."
    )
    try:
        tavily_api_key = st.secrets.get("TAVILY_API_KEY", "")
    except Exception:
        tavily_api_key = ""
    tavily_api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY", "")

    if not tavily_api_key:
        st.info(
            "To enable web search, create a Tavily API key and put it in "
            "`.streamlit/secrets.toml` as `TAVILY_API_KEY = \"your-key\"`, then restart HouseLens. "
            "The key stays in your local secrets file, which is excluded from Git."
        )

    with st.form("houselens_web_search"):
        web_query = st.text_input(
            "What do you want to search?",
            placeholder="For example: recent asking prices near Merelhoven 343",
        )
        run_search = st.form_submit_button("Search the web", type="primary", icon=":material/search:")

    if run_search:
        if not web_query.strip():
            st.warning("Enter a search phrase first.")
        elif not tavily_api_key:
            st.warning("Add your Tavily API key using the setup note above to run a web search.")
        else:
            try:
                with st.spinner("Searching the web…"):
                    st.session_state["houselens_web_results"] = search_web(web_query.strip(), tavily_api_key)
                    st.session_state["houselens_web_query"] = web_query.strip()
            except RuntimeError as error:
                st.error(str(error))

    search_results = st.session_state.get("houselens_web_results")
    if search_results:
        st.caption(f"Results for: {st.session_state.get('houselens_web_query', '')}")
        if search_results.get("answer"):
            st.markdown("**Search summary**")
            st.text(search_results["answer"])
        for result in search_results.get("results", []):
            with st.container(border=True):
                st.text(result.get("title", "Web result"))
                st.caption(result.get("content", ""))
                result_url = result.get("url", "")
                if isinstance(result_url, str) and result_url.startswith(("https://", "http://")):
                    st.link_button("Open source", result_url, icon=":material/open_in_new:")

st.divider()
st.caption("HouseLens supports comparison and learning. It does not decide whether a home is a good purchase. Prices are historical US dollars, and the model can be wrong, especially for unusual homes.")
