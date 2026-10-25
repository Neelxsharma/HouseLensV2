"""A small Streamlit demo for exploring the HouseLens model."""

from pathlib import Path
import json
import re
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).parent
MODEL_FILE = ROOT / "artifacts" / "house_price_model.joblib"
INFO_FILE = ROOT / "artifacts" / "model_info.json"
TREND_FILE = ROOT / "reports" / "price_trend.csv"
DATA_FILE = ROOT / "data" / "train.csv"
LISTINGS_FILE = ROOT / "data" / "listings.csv"
KLAVIDA_FEED_URL = "https://klavida.nl/open/aanbod.json"
KLAVIDA_CACHE_FILE = ROOT / "data" / "klavida_open_cache.json"
KLAVIDA_REFRESH_INTERVAL = timedelta(hours=1)
LISTING_COLUMNS = [
    "offer_type", "title", "city", "price", "area_m2", "rooms",
    "photo_url", "listing_url", "property_type", "energy_label", "source", "status",
]

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
def load_listings_file():
    if LISTINGS_FILE.exists() and LISTINGS_FILE.stat().st_size > 0:
        return pd.read_csv(LISTINGS_FILE)
    return pd.DataFrame(columns=LISTING_COLUMNS)


def read_klavida_cache():
    if not KLAVIDA_CACHE_FILE.exists():
        return None
    try:
        return json.loads(KLAVIDA_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
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
        "rooms": ["room_count", "number_of_rooms", "kamers"],
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
    frame["area_m2"] = frame["area_m2"].map(parse_number)
    frame["rooms"] = frame["rooms"].map(parse_number)
    frame["status"] = frame["status"].astype(str).str.strip().str.lower()
    inactive = {"sold", "rented", "let", "inactive", "removed", "verkocht", "verhuurd", "ingetrokken"}
    frame = frame[~frame["status"].isin(inactive)]
    return frame[frame["offer_type"].isin(["buy", "rent"])].copy()


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

with st.sidebar:
    st.header("Learning model · Ames, Iowa")
    st.write(f"Selected after testing: **{info['selected_model']}**")
    st.write(f"Test-set MAE: **${info['selected_model_test_mae_usd']:,.0f}**")
    st.caption("MAE is the average absolute difference between a prediction and a known sale price in the held-out test data. It is a rough guide, not a guaranteed error limit.")

tab_browse, tab_predict, tab_trends, tab_results = st.tabs([
    "Browse homes", "Price estimate demo", "Past prices demo", "Model results demo"
])

with tab_browse:
    st.subheader("Homes to buy" if housing_goal == "Buy" else "Homes to rent")
    st.caption(
        "Dutch listings with source links and listing photos. Rental prices are shown per month."
        if housing_goal == "Rent"
        else "Dutch listings with source links and listing photos. Asking prices are shown in euros."
    )

    klavida_cache = read_klavida_cache() or {}
    klavida_payload = klavida_cache.get("payload")

    with st.expander("Listing sources and CSV upload", icon=":material/add_home:"):
        if housing_goal == "Buy":
            st.markdown("**Klavida open sale listings**")
            st.caption(
                "Klavida allows reuse of listing facts with a link back. Its open feed does not include listing photos, "
                "and should be refreshed no more than once per hour."
            )
            last_fetch = klavida_cache.get("fetched_at")
            next_refresh = None
            if last_fetch:
                next_refresh = datetime.fromisoformat(last_fetch) + KLAVIDA_REFRESH_INTERVAL
            can_refresh = next_refresh is None or datetime.now(timezone.utc) >= next_refresh
            if st.button("Load or refresh Klavida homes", disabled=not can_refresh):
                with st.spinner("Loading current sale listings from Klavida…"):
                    ok, available_at, error = refresh_klavida_feed()
                if ok:
                    st.rerun()
                elif error:
                    st.warning(f"{error} Refresh available after {available_at.astimezone().strftime('%H:%M') }.")
            elif next_refresh and not can_refresh:
                st.caption(f"Next refresh available after {next_refresh.astimezone().strftime('%H:%M')}.")
            if klavida_cache.get("error"):
                st.caption(f"Last feed update issue: {klavida_cache['error']}")
        else:
            st.markdown("**Rental listings**")
            st.caption("Klavida’s open feed is for sale listings. Rental homes need a separate permitted rental feed or CSV.")

        st.write(
            "You can also add a provider's permitted export as `data/listings.csv`, or upload a CSV here. "
            "Include `photo_url` and `listing_url` so each card can show the original listing photo and link."
        )
        st.caption(
            "Required fields: offer_type (buy/rent), price, city, listing_url. "
            "Useful extras: title, photo_url, area_m2, rooms, property_type, energy_label, source, status."
        )
        template = pd.DataFrame(columns=LISTING_COLUMNS)
        st.download_button(
            "Download CSV template",
            template.to_csv(index=False).encode("utf-8"),
            file_name="houselens_listings_template.csv",
            mime="text/csv",
            icon=":material/download:",
        )
        uploaded_listings = st.file_uploader(
            "Upload a listings CSV",
            type=["csv"],
            help="Uploads are used for this session only. The CSV should contain listing data you have permission to reuse.",
        )

    if uploaded_listings is not None:
        try:
            raw_listings = pd.read_csv(uploaded_listings)
        except pd.errors.EmptyDataError:
            st.warning("That CSV is empty. Add listing rows and upload it again.")
            raw_listings = pd.DataFrame(columns=LISTING_COLUMNS)
    else:
        raw_listings = load_listings_file()
        if raw_listings.empty and housing_goal == "Buy" and klavida_payload is not None:
            raw_listings = klavida_listings(klavida_payload)
    listings = normalize_listings(raw_listings)
    listings = listings[listings["offer_type"] == ("buy" if housing_goal == "Buy" else "rent")]

    if listings.empty:
        with st.container(border=True):
            if housing_goal == "Rent":
                st.subheader("Rental listings need a separate source")
                st.write("Klavida provides reusable sale listing facts, not rental listings. Add a permitted rental feed or upload its CSV export here.")
            elif klavida_payload is not None:
                st.subheader("Klavida is connected, but no listings were read")
                st.write("The feed response did not contain sale listings in a format HouseLens could display. You can still upload an authorised CSV export here.")
            else:
                st.subheader("Klavida sale listings are ready to load")
                st.write("Load the current Klavida open feed above or upload its data. Each result links back to Klavida; photos are not included in its feed.")
            st.caption(
                "No sample properties or stock photos are shown as real listings. Use only feeds and photos you have permission to display."
            )
    else:
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
        if selected_city != "All locations":
            filtered = filtered[filtered["city"].astype(str) == selected_city]
        if price_limit:
            filtered = filtered[filtered["price"].fillna(0) <= price_limit]
        if minimum_rooms != "Any":
            filtered = filtered[filtered["rooms"].fillna(0) >= minimum_rooms]
        st.caption(f"{len(filtered)} homes found · Listing photos and availability come from the source feed.")

        if filtered.empty:
            st.info("No homes match these filters. Try a higher budget or a different location.")
        else:
            cards = st.columns(3)
            for index, (_, listing) in enumerate(filtered.iterrows()):
                with cards[index % len(cards)]:
                    with st.container(border=True):
                        photo = str(listing["photo_url"]).strip()
                        if photo and photo.lower() not in {"nan", "none"}:
                            st.image(photo, width="stretch")
                        else:
                            st.info("Photo not supplied by listing source", icon=":material/image:")
                        st.badge("For sale" if housing_goal == "Buy" else "For rent", color="blue")
                        st.markdown(f"**{listing['title']}**")
                        price_text = f"€{listing['price']:,.0f}"
                        if housing_goal == "Rent":
                            price_text += " / month"
                        st.markdown(f"### {price_text}")
                        details = [str(listing["city"]), str(listing["property_type"])]
                        if pd.notna(listing["area_m2"]):
                            details.append(f"{listing['area_m2']:g} m²")
                        if pd.notna(listing["rooms"]):
                            details.append(f"{listing['rooms']:g} rooms")
                        if str(listing["energy_label"]).strip():
                            details.append(f"Label {listing['energy_label']}")
                        st.caption(" · ".join(value for value in details if value and value.lower() != "nan"))
                        st.caption(f"Source: {listing['source']}")
                        listing_url = str(listing["listing_url"]).strip()
                        if listing_url and listing_url.lower() not in {"nan", "none"}:
                            st.link_button("View original listing", listing_url, width="stretch", icon=":material/open_in_new:")

with tab_predict:
    st.subheader("Enter a few house details")
    st.write("Change the values to see how the estimate responds. The model only knows the features listed here.")
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

    if st.button("Estimate price", type="primary"):
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
        st.caption("The model learned from past Ames sales. A current asking price, renovation details, interest rates, and many other factors are not included.")

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

st.divider()
st.caption("HouseLens supports comparison and learning. It does not decide whether a home is a good purchase. Prices are historical US dollars, and the model can be wrong, especially for unusual homes.")
