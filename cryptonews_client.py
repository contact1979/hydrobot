# /data_collection/cryptonews_client.py

import logging
import requests
import datetime
import pytz # For timezone handling
import pandas as pd
from typing import List, Dict, Optional, Tuple
import time # Import time for potential delays
import json # Import json for serialization if needed

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config
from database import db_utils # Import needed for JSON_TYPE check
# from .. import config # Use relative import if running as part of a package
# from ..database import db_utils

log = logging.getLogger(__name__)

# --- CryptoNews API Configuration ---
CRYPTONEWS_API_TOKEN = config.CRYPTONEWS_API_TOKEN
CRYPTONEWS_BASE_URL = "https://cryptonews-api.com/api/v1"

# Define the source timezone (Eastern Time) - Using pytz for DST handling
SOURCE_TIMEZONE_STR = 'US/Eastern'
try:
    SOURCE_TIMEZONE = pytz.timezone(SOURCE_TIMEZONE_STR)
except pytz.exceptions.UnknownTimeZoneError:
    log.error(f"Unknown timezone '{SOURCE_TIMEZONE_STR}'. Using UTC as fallback for ET conversion.")
    SOURCE_TIMEZONE = pytz.utc # Fallback, though calculations might be off

TARGET_TIMEZONE = pytz.utc # Store everything in UTC

def _parse_cryptonews_date(date_str: Optional[str]) -> Optional[datetime.datetime]:
    """Parses date string from API (assuming various formats) and converts to UTC."""
    if not date_str:
        return None
    try:
        # Try parsing formats with timezone offset first (more reliable)
        # Example: "Fri, 08 Mar 2024 14:01:09 -0500"
        # Pandas to_datetime is quite flexible
        dt_aware = pd.to_datetime(date_str, utc=False) # Parse as naive first or aware if offset exists
        if dt_aware.tzinfo:
            # If pandas parsed timezone info, convert directly to UTC
            return dt_aware.tz_convert(TARGET_TIMEZONE)
        else:
            # If naive, assume it's in the SOURCE_TIMEZONE (e.g., ET) and convert
            dt_aware_source = SOURCE_TIMEZONE.localize(dt_aware)
            return dt_aware_source.astimezone(TARGET_TIMEZONE)
    except Exception as e:
        log.warning(f"Could not parse CryptoNews date '{date_str}': {e}. Returning None.")
        return None

def _make_api_request(endpoint_path: str, params: Dict) -> Optional[Dict]:
    """Helper function to make requests and handle common errors."""
    if not CRYPTONEWS_API_TOKEN:
        log.error("CryptoNews API Token is not configured.")
        return None

    # Ensure token is always included
    params['token'] = CRYPTONEWS_API_TOKEN
    url = f"{CRYPTONEWS_BASE_URL}{endpoint_path}"

    log.debug(f"Making CryptoNews API request to {url} with params: {params}")
    try:
        response = requests.get(url, params=params, timeout=30) # Generous timeout
        log.debug(f"CryptoNews API response status: {response.status_code}")
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if 'error' in data and data['error']:
             log.error(f"CryptoNews API returned an error for {url} with params {params}: {data['error']}")
             return None
        # Allow responses that might not have a 'data' key (e.g., if endpoint returns directly)
        # if 'data' not in data:
        #      log.warning(f"No 'data' key found in CryptoNews API response for {url}. Response: {data}")
        #      return None # Or return empty data structure?

        return data

    except requests.exceptions.Timeout:
        log.error(f"Timeout error fetching data from CryptoNews API: {url}")
    except requests.exceptions.RequestException as e:
        log.error(f"Request error fetching news from CryptoNews API: {e}", exc_info=False) # Reduce noise
        if e.response is not None:
            log.error(f"Response status: {e.response.status_code}, Body: {e.response.text[:500]}...")
    except Exception as e:
        log.error(f"Unexpected error processing CryptoNews request or response: {e}", exc_info=True)

    return None

def _process_news_articles(raw_articles: List[Dict], source_api_name: str = 'cryptonews') -> List[Dict]:
    """Processes a list of raw article dicts from the API response."""
    processed_items = []
    if not isinstance(raw_articles, list):
        log.warning(f"Expected a list of articles, but got {type(raw_articles)}. Skipping processing.")
        return []

    for article in raw_articles:
        try:
            published_at_utc = _parse_cryptonews_date(article.get('date'))
            url = article.get('news_url')

            if url and published_at_utc: # Require URL and valid date
                tickers = article.get('tickers')
                # Ensure tickers is stored appropriately for DB (JSON string or list)
                if isinstance(tickers, list) and db_utils.JSON_TYPE:
                    tickers_stored = tickers # Store as list if DB supports JSON
                elif isinstance(tickers, list):
                    tickers_stored = json.dumps(tickers) # Convert list to JSON string for TEXT columns
                else:
                    tickers_stored = None # Or handle other types if necessary

                processed_items.append({
                    'source_api': source_api_name,
                    'source_publisher': article.get('source_name'),
                    'article_id': None, # Not obviously available in examples
                    'title': article.get('title'),
                    'text_content': article.get('text'),
                    'url': url,
                    'published_at': published_at_utc, # Store converted UTC datetime
                    'tickers_mentioned': tickers_stored,
                })
            else:
                 log.debug(f"Skipping article due to missing URL or unparseable date: {url}, {article.get('date')}")
        except Exception as item_err:
            log.error(f"Error processing individual news article: {item_err}. Article data: {article}", exc_info=True)
    return processed_items


def fetch_ticker_news(tickers: List[str], items_per_page: int = 50, page: int = 1) -> List[Dict]:
    """
    Fetches the *latest* news for specific tickers from the CryptoNews API.
    Uses the default endpoint which implies /posts or similar based on examples.

    Args:
        tickers (List[str]): A list of ticker symbols (e.g., ['BTC', 'ETH']).
        items_per_page (int): Number of news items to fetch (max 50-100, check docs).
        page (int): Page number for pagination.

    Returns:
        List[Dict]: A list of processed news article dictionaries. Empty list on failure.
    """
    if not tickers:
        log.warning("No tickers provided to fetch news for.")
        return []

    # Assuming '/posts' or just the base URL works for ticker filtering based on docs
    # If docs specify '/category?section=alltickers&tickers=...', adjust endpoint_path
    endpoint_path = ""
    ticker_string = ",".join(tickers).upper()
    params = {
        "tickers": ticker_string,
        "items": min(items_per_page, 100), # Use 100 as a safe max, adjust if known
        "page": page,
        # No date parameter means fetch latest
    }

    log.info(f"Fetching LATEST CryptoNews for tickers: {ticker_string}, page: {page}, items: {params['items']}")
    data = _make_api_request(endpoint_path, params)

    if data and isinstance(data.get('data'), list):
        return _process_news_articles(data['data'])
    else:
        log.debug(f"No valid 'data' list found in response for latest news fetch: {ticker_string}")
        return []


def fetch_historical_ticker_news(tickers: List[str], date_str: str, items_per_page: int = 50, page: int = 1) -> Tuple[List[Dict], bool]:
    """
    Fetches HISTORICAL news for specific tickers from the CryptoNews API using the date parameter.

    Args:
        tickers (List[str]): A list of ticker symbols (e.g., ['BTC', 'ETH']).
        date_str (str): Date range string in API format (e.g., 'MMDDYYYY-MMDDYYYY' or 'last7days').
        items_per_page (int): Number of news items to fetch (max 50-100).
        page (int): Page number for pagination.

    Returns:
        Tuple[List[Dict], bool]:
            - A list of processed news article dictionaries. Empty list on failure.
            - Boolean indicating if more pages might be available (True if items_per_page == returned count, False otherwise).
    """
    if not tickers:
        log.warning("No tickers provided to fetch historical news for.")
        return [], False
    if not date_str:
         log.warning("Date string parameter is required for historical fetching.")
         return [], False

    # Assuming base URL + params works for ticker filtering based on docs
    endpoint_path = ""
    ticker_string = ",".join(tickers).upper()
    params = {
        "tickers": ticker_string,
        "items": min(items_per_page, 100), # Use 100 as a safe max
        "page": page,
        "date": date_str # Add the date parameter
    }

    log.info(f"Fetching HISTORICAL CryptoNews for tickers: {ticker_string}, Date: {date_str}, page: {page}, items: {params['items']}")
    data = _make_api_request(endpoint_path, params)
    news_items = []
    has_more_pages = False

    if data and isinstance(data.get('data'), list):
        raw_articles = data['data']
        log.info(f"Received {len(raw_articles)} historical news items from CryptoNews API for tickers {ticker_string}, date {date_str}, page {page}.")
        news_items = _process_news_articles(raw_articles)
        # Check if the number of items returned equals the number requested, suggesting more pages might exist
        if len(raw_articles) == params['items']:
            # Basic plans might be limited to 5 pages total, check API docs/response if possible
            if page < 5: # Crude check based on docs mentioning 5 page limit for basic
                 has_more_pages = True
            else:
                 log.warning("Reached page 5, assuming no more pages due to potential basic plan limit.")

    else:
         log.debug(f"No valid 'data' list found in response for historical news fetch: {ticker_string}, date {date_str}, page {page}")

    return news_items, has_more_pages


# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # Setup basic logging for direct script run
    logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s [%(name)s:%(lineno)d] %(message)s')

    print("--- Testing CryptoNews Client ---")

    if not config.CRYPTONEWS_API_TOKEN:
         print("\nERROR: CryptoNews API Token (CRYPTONEWS_API_TOKEN) not configured in .env")
    else:
        test_tickers = ['BTC', 'ETH']

        # Test fetching latest news
        print(f"\nFetching latest news for tickers: {test_tickers}...")
        latest_news = fetch_ticker_news(tickers=test_tickers, items_per_page=5)
        if latest_news:
            print(f"Fetched {len(latest_news)} latest news items.")
            if latest_news:
                 print("First item details:")
                 print(latest_news[0])
        else:
            print("Could not fetch latest news or no news found.")

        print("\n" + "="*20 + "\n")

        # Test fetching historical news (last 7 days)
        print(f"\nFetching historical news for tickers: {test_tickers} (last7days)...")
        page = 1
        all_historical = []
        while True:
            historical_news_page, more_pages = fetch_historical_ticker_news(
                tickers=test_tickers,
                date_str='last7days', # Use the API's relative date string
                items_per_page=5, # Small number for testing pagination
                page=page
            )
            if historical_news_page:
                all_historical.extend(historical_news_page)
                if more_pages:
                     page += 1
                     print(f"Fetching next page ({page})...")
                     time.sleep(1) # Delay between pages
                else:
                     print("No more pages indicated.")
                     break
            else:
                 print(f"Could not fetch historical news on page {page} or no news found.")
                 break

        if all_historical:
            print(f"Fetched {len(all_historical)} total historical news items.")
            print("First historical item details:")
            print(all_historical[0])
        else:
             print("No historical news fetched.")


    print("\n--- Test Complete ---")
