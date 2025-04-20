# /data_collection/reddit_client.py

import logging
import praw
import datetime
import pytz
from prawcore.exceptions import ResponseException, RequestException, PrawcoreException
from typing import List, Dict

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config
# from .. import config # Use relative import if running as part of a package

log = logging.getLogger(__name__)

# --- Reddit Client Initialization ---
_reddit_client = None

def get_reddit_client():
    """Initializes and returns the PRAW Reddit instance singleton."""
    global _reddit_client
    if _reddit_client is None:
        client_id = config.REDDIT_CLIENT_ID
        client_secret = config.REDDIT_CLIENT_SECRET
        user_agent = config.REDDIT_USER_AGENT
        # Optional: Add username/password if needed for specific actions, but read-only is usually sufficient
        # username = config.REDDIT_USERNAME
        # password = config.REDDIT_PASSWORD

        if not all([client_id, client_secret, user_agent]):
            log.error("Reddit API credentials (client ID, secret, user agent) are not fully configured.")
            return None

        try:
            log.info("Initializing PRAW Reddit instance...")
            _reddit_client = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
                # username=username, # Uncomment if using username/password
                # password=password, # Uncomment if using username/password
                read_only=True # Set to False if write actions are needed
            )
            # Basic check to see if authentication worked (optional)
            # log.info(f"Reddit client authenticated as: {_reddit_client.user.me()}") # Requires read_only=False or OAuth
            log.info("PRAW Reddit instance initialized successfully (read_only mode).")
        except PrawcoreException as e:
            log.error(f"PRAW Error initializing Reddit instance: {e}", exc_info=True)
            _reddit_client = None
        except Exception as e:
            log.error(f"Unexpected error initializing PRAW Reddit instance: {e}", exc_info=True)
            _reddit_client = None
    return _reddit_client

def fetch_new_subreddit_posts(subreddit_names: List[str], post_limit_per_subreddit: int = config.REDDIT_POST_LIMIT) -> List[Dict]:
    """
    Fetches the newest posts (submissions) from specified subreddits.

    Args:
        subreddit_names (list): A list of subreddit names (without 'r/').
        post_limit_per_subreddit (int): Max number of posts to fetch from the 'new' section
                                         of each subreddit per call.

    Returns:
        list: A list of dictionaries representing posts. Empty list on failure.
              Includes 'created_utc_dt' key with timezone-aware UTC datetime.
    """
    reddit = get_reddit_client()
    if reddit is None:
        log.error("Cannot fetch Reddit posts, PRAW instance not available.")
        return []

    reddit_posts = []
    if not isinstance(subreddit_names, list):
        log.error("Subreddits parameter must be a list.")
        return []

    log.info(f"Fetching up to {post_limit_per_subreddit} new posts from subreddits: {subreddit_names}")

    for sub_name in subreddit_names:
        try:
            log.debug(f"Accessing subreddit: r/{sub_name}")
            subreddit = reddit.subreddit(sub_name)
            # Fetch newest posts using .new()
            new_posts = subreddit.new(limit=post_limit_per_subreddit)

            count = 0
            for post in new_posts:
                # Skip stickied posts if desired (often mod announcements)
                if post.stickied:
                    log.debug(f"Skipping stickied post in r/{sub_name}: {post.id}")
                    continue

                # Combine title and selftext for analysis later
                post_text_combined = post.title
                if post.is_self and post.selftext:
                    post_text_combined += "\n" + post.selftext

                # Convert timestamp to timezone-aware datetime object
                created_utc_dt = datetime.datetime.fromtimestamp(post.created_utc, tz=pytz.utc)

                reddit_posts.append({
                    'post_id': post.id,
                    'subreddit': sub_name.lower(), # Store lowercase for consistency
                    'title': post.title,
                    'selftext': post.selftext if post.is_self else None,
                    'text_combined': post_text_combined, # For easier sentiment analysis input
                    'url': f"https://www.reddit.com{post.permalink}",
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc_dt': created_utc_dt, # Store as datetime object
                    # 'fetched_at' will be added by the database insertion logic
                })
                count += 1
            log.info(f"Fetched {count} posts from r/{sub_name}")

        except ResponseException as e:
            # Handle HTTP errors (e.g., 404 Not Found, 403 Forbidden, 5xx Server Errors)
            log.error(f"PRAW HTTP error fetching from r/{sub_name}: {e.response.status_code} - {e}")
        except RequestException as e:
             # Handle network-related errors (e.g., connection timeout)
             log.error(f"PRAW request error fetching from r/{sub_name}: {e}")
        except PrawcoreException as e:
             # Handle other PRAW-specific errors
             log.error(f"PRAW core error processing subreddit r/{sub_name}: {e}", exc_info=True)
        except Exception as e:
            # Catch any other unexpected errors
            log.error(f"General error processing subreddit r/{sub_name}: {e}", exc_info=True)

    log.info(f"Finished fetching Reddit posts. Total items collected: {len(reddit_posts)}")
    return reddit_posts


# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # Setup basic logging for direct script run
    logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s [%(name)s] %(message)s')

    print("--- Testing Reddit Client ---")

    # Check credentials loaded via config
    if not all([config.REDDIT_CLIENT_ID, config.REDDIT_CLIENT_SECRET, config.REDDIT_USER_AGENT]):
         print("\nERROR: Reddit API credentials not fully configured in .env")
    else:
        target_subs = config.TARGET_SUBREDDITS[:2] # Test with first 2 configured subreddits
        print(f"\nFetching latest posts from subreddits: {target_subs}...")

        posts_list = fetch_new_subreddit_posts(subreddit_names=target_subs, post_limit_per_subreddit=5)

        if posts_list:
            print(f"Fetched {len(posts_list)} total posts.")
            print("\nFirst post details:")
            first_post = posts_list[0]
            for key, value in first_post.items():
                 # Truncate long text fields for display
                 if key in ['selftext', 'text_combined'] and value and len(value) > 100:
                     print(f"  {key}: {value[:100]}...")
                 else:
                     print(f"  {key}: {value}")
            # Verify UTC conversion
            if first_post.get('created_utc_dt'):
                print(f"  Created (UTC): {first_post['created_utc_dt']}")
                print(f"  Is UTC Timezone Aware: {first_post['created_utc_dt'].tzinfo is not None}")
        else:
            print("Could not fetch any Reddit posts. Check logs for errors.")

    print("\n--- Test Complete ---")

