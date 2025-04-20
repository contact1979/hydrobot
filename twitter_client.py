# /data_collection/twitter_client.py

import logging
import tweepy
import datetime
import pytz
from typing import List, Dict, Optional, Tuple
import time
import json # Added for JSON serialization if needed

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config
# from .. import config # Use relative import if running as part of a package

log = logging.getLogger(__name__)

# --- Twitter Client Initialization ---
_twitter_client_v2 = None

def get_twitter_client_v2():
    """Initializes and returns the Tweepy API v2 client singleton."""
    global _twitter_client_v2
    if _twitter_client_v2 is None:
        bearer_token = config.TWITTER_BEARER_TOKEN
        if not bearer_token:
            log.error("Twitter API v2 Bearer Token (TWITTER_BEARER_TOKEN) not configured.")
            return None
        try:
            log.info("Initializing Tweepy client for Twitter API v2...")
            # Use bearer token for app-only authentication (suitable for searching tweets)
            _twitter_client_v2 = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=False) # Handle rate limits manually for logging
            # Test authentication
            # Note: get_me() requires User context (OAuth 2 PKCE or OAuth 1), not available with App-only Bearer Token.
            # We can just assume initialization works if no exception occurs.
            # me_response = _twitter_client_v2.get_me() # This will fail with Bearer Token
            # Instead, maybe try a simple search? Or just proceed.
            log.info("Tweepy client object created successfully (using Bearer Token).")

        except tweepy.errors.TweepyException as e:
            log.error(f"TweepyException initializing Twitter client: {e}", exc_info=True)
            _twitter_client_v2 = None
        except Exception as e:
            log.error(f"Unexpected error initializing Twitter client: {e}", exc_info=True)
            _twitter_client_v2 = None
    return _twitter_client_v2

def build_twitter_query(symbols: List[str], base_keywords: List[str] = config.TWITTER_QUERY_KEYWORDS) -> str:
    """
    Builds a search query string for the Twitter API v2 Recent Search endpoint.
    Focuses on cashtags and hashtags for the given symbols, combined with base keywords.
    Excludes retweets and requires English language.

    Args:
        symbols (List[str]): List of cryptocurrency symbols (e.g., ['BTC', 'ETH']).
                             Should be just the base symbol, without 'USDT'.
        base_keywords (List[str]): General keywords to include (e.g., ['crypto']).

    Returns:
        str: A formatted query string for the Twitter API.
             Returns an empty string if no symbols or keywords provided.
    """
    if not symbols and not base_keywords:
        log.warning("Cannot build Twitter query without symbols or keywords.")
        return ""

    symbol_query_parts = []
    if symbols:
        # Create cashtag/hashtag parts for each symbol
        symbol_parts = []
        for symbol in symbols:
            s_upper = symbol.upper().replace('USDT', '').replace('BUSD','') # Ensure base symbol, uppercase
            if s_upper:
                symbol_parts.append(f"#{s_upper}")  # Remove cashtag ($) operator
        if symbol_parts:
            symbol_query_parts.append(f"({' OR '.join(symbol_parts)})")

    keyword_query_parts = []
    if base_keywords:
        keyword_part = " ".join([f'"{k}"' if ' ' in k else k for k in base_keywords]) # Quote keywords with spaces
        keyword_query_parts.append(f"({keyword_part})")

    # Combine symbol and keyword parts (if both exist, combine with OR or AND depending on need - using OR for broader reach)
    combined_terms = " OR ".join(symbol_query_parts + keyword_query_parts)
    if not combined_terms:
         log.warning("No valid symbols or keywords resulted in query terms.")
         return ""

    # Combine parts, add language filter and exclude retweets
    full_query = f"({combined_terms}) lang:en -is:retweet"

    # Twitter query length limit is 512 chars for standard v2 basic/elevated, 1024 for academic
    max_len = 512
    if len(full_query) > max_len:
        log.warning(f"Generated Twitter query exceeds max length ({max_len}). Truncating: {full_query}")
        # Simple truncation, might break logic. Consider smarter query splitting if this happens often.
        full_query = full_query[:max_len].rsplit(' ', 1)[0] # Try to cut at last space
        log.warning(f"Truncated query: {full_query}")

    log.debug(f"Generated Twitter Query: {full_query}")
    return full_query.strip()


def search_recent_tweets(query: str, max_total_results: int = 100, results_per_page: int = 100) -> Tuple[List[Dict], Optional[str]]:
    """
    Searches for recent tweets matching the query using Twitter API v2 (handles pagination).

    Args:
        query (str): The search query string.
        max_total_results (int): The maximum number of tweets to attempt to fetch across all pages.
                                 Limited by API tier limits (e.g., 100 for basic recent search in total usually).
        results_per_page (int): Max number of tweets per API request (10-100 for recent search).

    Returns:
        Tuple[List[Dict], Optional[str]]:
            - A list of dictionaries representing tweets. Empty list on failure.
              Includes 'created_at_utc' key with timezone-aware UTC datetime.
            - The 'newest_id' from the metadata of the *last successful* response,
              suitable for use as 'since_id' in the next poll. Returns None if no tweets found or error.
    """
    client = get_twitter_client_v2()
    if not client:
        log.error("Cannot search tweets, Twitter client not available.")
        return [], None
    if not query:
        log.warning("Empty query provided, skipping Twitter search.")
        return [], None

    log.info(f"Searching recent tweets with query: '{query}', aiming for max {max_total_results} results...")

    all_tweets_list = []
    newest_id_overall = None
    next_page_token = None
    fetched_count = 0
    max_pages_to_fetch = (max_total_results + results_per_page - 1) // results_per_page # Calculate max pages needed

    # Define the fields we want to retrieve for each tweet
    tweet_fields = ["created_at", "public_metrics", "entities", "author_id"]

    backoff_time = 1  # Initial backoff time in seconds

    for page_num in range(max_pages_to_fetch):
        if fetched_count >= max_total_results:
            log.info(f"Reached max_total_results limit ({max_total_results}). Stopping pagination.")
            break

        page_max_results = min(results_per_page, 100, max_total_results - fetched_count) # API max is 100
        if page_max_results < 10: page_max_results = 10 # API min is 10

        log.debug(f"Fetching Twitter page {page_num + 1}, max_results={page_max_results}, next_token={next_page_token}")

        try:
            response = client.search_recent_tweets(
                query=query,
                max_results=page_max_results,
                tweet_fields=tweet_fields,
                next_token=next_page_token
            )

            # Handle potential errors in response
            if response.errors:
                log.error(f"Twitter API returned errors on page {page_num + 1}: {response.errors}")
                # Check for rate limit errors specifically
                is_rate_limit = any(error.get("code") == 429 or error.get("status") == 429 for error in response.errors)
                if is_rate_limit:
                    log.warning("Twitter API rate limit hit (429). Retrying after backoff.")
                    time.sleep(backoff_time)
                    backoff_time = min(backoff_time * 2, 300)  # Exponential backoff, max 5 minutes
                    continue
                break  # Stop pagination on other errors

            current_page_tweets = []
            if response.data:
                log.info(f"Received {len(response.data)} tweets on page {page_num + 1}.")
                for tweet in response.data:
                    try:
                        # Extract entities (hashtags, cashtags)
                        hashtags = [tag['tag'] for tag in tweet.entities.get('hashtags', [])] if tweet.entities else []
                        cashtags = [tag['tag'] for tag in tweet.entities.get('cashtags', [])] if tweet.entities else []

                        # Convert created_at to timezone-aware UTC datetime
                        created_at_utc = tweet.created_at.replace(tzinfo=pytz.utc) if tweet.created_at else None

                        current_page_tweets.append({
                            'tweet_id': str(tweet.id),
                            'author_id': str(tweet.author_id) if tweet.author_id else None,
                            'text': tweet.text,
                            'created_at': created_at_utc,
                            'public_metrics': tweet.public_metrics or {},
                            'hashtags': hashtags,
                            'cashtags': cashtags,
                        })
                    except Exception as item_err:
                        log.error(f"Error processing individual tweet: {item_err}. Tweet data: {tweet}", exc_info=True)

                all_tweets_list.extend(current_page_tweets)
                fetched_count += len(current_page_tweets)

                # Update newest_id from the *first* page's metadata
                if page_num == 0 and response.meta and 'newest_id' in response.meta:
                    newest_id_overall = response.meta['newest_id']
                    log.debug(f"Newest tweet ID from first page: {newest_id_overall}")

            # Check for next page token
            if response.meta and 'next_token' in response.meta:
                next_page_token = response.meta['next_token']
                log.debug("Found next_token, will fetch next page.")
                time.sleep(1.1)  # Add delay between paginated requests
            else:
                log.debug("No next_token found, pagination complete for this query.")
                break  # Exit pagination loop

        except tweepy.errors.TweepyException as e:
            if isinstance(e, tweepy.errors.HTTPException) and e.response and e.response.status_code == 429:
                log.warning("Twitter API rate limit hit (429 Exception). Retrying after backoff.")
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, 300)  # Exponential backoff, max 5 minutes
                continue
            log.error(f"TweepyException searching tweets (Page {page_num + 1}): {e}", exc_info=True)
            break  # Stop pagination on other errors
        except Exception as e:
            log.error(f"Unexpected error searching tweets (Page {page_num + 1}): {e}", exc_info=True)
            break  # Stop pagination on error

    log.info(f"Finished Twitter search for query. Total tweets fetched: {len(all_tweets_list)}")
    return all_tweets_list, newest_id_overall


# --- Example Usage (for testing) ---
# Store the last fetched tweet ID for pagination testing
# In a real app, this state needs to be persisted (e.g., in DB or a state file)
_LATEST_TWEET_ID_STORE_TEST = {}

if __name__ == '__main__':
    # Setup basic logging for direct script run
    logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s [%(name)s:%(lineno)d] %(message)s')

    print("--- Testing Twitter Client ---")

    if not config.TWITTER_BEARER_TOKEN:
         print("\nERROR: Twitter Bearer Token (TWITTER_BEARER_TOKEN) not configured in .env")
    else:
        # Test with a few symbols
        test_symbols = ['BTC', 'ETH'] # Use symbols likely to have recent tweets
        test_query = build_twitter_query(symbols=test_symbols, base_keywords=['price'])

        if test_query:
            print(f"\nGenerated Query: {test_query}")

            # --- Simulate polling with since_id ---
            query_key_test = "test_btc_eth"
            # 1. First fetch
            print("\n--- Fetch 1 (No Since ID) ---")
            since_id_for_run = _LATEST_TWEET_ID_STORE_TEST.get(query_key_test)
            tweets1, newest_id1 = search_recent_tweets(query=test_query, max_total_results=20, results_per_page=10) # Fetch max 20 in 2 pages
            if tweets1:
                print(f"Fetched {len(tweets1)} tweets.")
                print("First tweet:", tweets1[0]['text'][:100] + "...")
                print(f"Newest ID: {newest_id1}")
                if newest_id1:
                     _LATEST_TWEET_ID_STORE_TEST[query_key_test] = newest_id1 # Update store
            else:
                print("No new tweets found in Fetch 1.")

            # Add a small delay before next fetch if testing pagination
            print("\nWaiting 5 seconds...")
            time.sleep(5)

            # 2. Second fetch (should ideally fetch 0 if no new tweets in 5s)
            print("\n--- Fetch 2 (Using Since ID from Fetch 1) ---")
            since_id_for_run = _LATEST_TWEET_ID_STORE_TEST.get(query_key_test) # Get the updated ID
            print(f"Using since_id: {since_id_for_run}")
            tweets2, newest_id2 = search_recent_tweets(query=test_query, max_total_results=20, results_per_page=10, since_id=since_id_for_run)
            if tweets2:
                print(f"Fetched {len(tweets2)} tweets.") # Should likely be 0 or very few
                if newest_id2:
                     _LATEST_TWEET_ID_STORE_TEST[query_key_test] = newest_id2
            else:
                print("No new tweets found in Fetch 2 (as expected potentially).")
                print(f"Newest ID from Fetch 2 meta (if any): {newest_id2}")

        else:
            print("Could not generate a valid test query.")

    print("\n--- Test Complete ---")

