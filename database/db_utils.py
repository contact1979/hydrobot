# /database/db_utils.py

import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Text, JSON, UniqueConstraint, Index, inspect
from sqlalchemy.dialects.postgresql import JSONB # Example for PostgreSQL JSON
# from sqlalchemy.dialects.mssql import JSON # Example for MS SQL Server JSON
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from contextlib import contextmanager
from typing import Optional # Added for type hinting

# Use absolute imports for the new structure
from hydrobot.config.settings import settings
from hydrobot.utils.logger_setup import get_logger

# Use the configured logger
log = get_logger(__name__)

# --- Database Setup ---
DATABASE_URL: Optional[str] = None
engine = None
SessionLocal = None
metadata = MetaData()

# Get DATABASE_URL from settings
db_settings = getattr(settings, 'database', None)
if db_settings:
    DATABASE_URL = getattr(db_settings, 'url', None)

if DATABASE_URL:
    try:
        # Adjust connect_args based on database type if needed
        # Example for SQLite (testing): engine = create_engine("sqlite:///./test.db", connect_args={"check_same_thread": False})
        engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        # Log partial URL for security - mask credentials if present
        safe_url = DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL
        log.info(f"Database engine created for URL ending with: ...{safe_url[-50:]}")
    except Exception as e:
        log.error(f"Failed to create database engine or session maker: {e}", exc_info=True)
        # Application might need to exit or run in a limited mode
else:
    log.critical("DATABASE_URL not configured in settings.database.url. Database features will be unavailable.")
    # Application might need to exit or run in a limited mode


# --- Define Database Tables ---
# Use JSONB for PostgreSQL, adjust for other DBs (e.g., Text or specific JSON type)
JSON_TYPE = JSONB if engine and engine.dialect.name == 'postgresql' else JSON

price_data = Table(
    'price_data', metadata,
    Column('id', Integer, primary_key=True),
    Column('symbol', String(32), nullable=False),
    Column('interval', String(8), nullable=False), # e.g., '5m'
    Column('open_time', DateTime(timezone=True), nullable=False),
    Column('open', Float, nullable=False),
    Column('high', Float, nullable=False),
    Column('low', Float, nullable=False),
    Column('close', Float, nullable=False),
    Column('volume', Float, nullable=False),
    Column('close_time', DateTime(timezone=True), nullable=False),
    # Technical Indicators - Adjust names based on actual calculation output
    Column('sma_fast', Float, nullable=True),
    Column('sma_slow', Float, nullable=True),
    Column('ema_fast', Float, nullable=True),
    Column('ema_slow', Float, nullable=True),
    Column('rsi_value', Float, nullable=True),
    Column('macd_line', Float, nullable=True),
    Column('macd_signal', Float, nullable=True),
    Column('macd_hist', Float, nullable=True),
    Column('fetched_at', DateTime(timezone=True), server_default=func.now()),
    # Constraints and Indexes
    UniqueConstraint('symbol', 'interval', 'open_time', name='uq_price_data_symbol_interval_time'),
    Index('ix_price_data_symbol_interval_open_time', 'symbol', 'interval', 'open_time'),
    Index('ix_price_data_open_time', 'open_time') # Index for time-based queries
)

news_data = Table(
    'news_data', metadata,
    Column('id', Integer, primary_key=True),
    Column('source_api', String(50), nullable=False), # e.g., 'cryptonews'
    Column('source_publisher', String(100), nullable=True), # e.g., 'CoinDesk'
    Column('article_id', String(255), nullable=True), # Unique ID from source API
    Column('title', String(512), nullable=False),
    Column('text_content', Text, nullable=True),
    Column('url', String(1024), nullable=False, unique=True),
    Column('published_at', DateTime(timezone=True), nullable=False), # Stored in UTC
    Column('tickers_mentioned', JSON_TYPE, nullable=True), # Store as list ['BTC', 'ETH']
    Column('fetched_at', DateTime(timezone=True), server_default=func.now()),
    # Indexes
    Index('ix_news_data_published_at', 'published_at'),
    Index('ix_news_data_source_api', 'source_api')
)

reddit_data = Table(
    'reddit_data', metadata,
    Column('id', Integer, primary_key=True),
    Column('post_id', String(20), nullable=False, unique=True), # Reddit Post ID
    Column('subreddit', String(100), nullable=False),
    Column('title', String(512), nullable=False),
    Column('selftext', Text, nullable=True),
    Column('url', String(1024), nullable=False),
    Column('score', Integer, nullable=True),
    Column('num_comments', Integer, nullable=True),
    Column('created_utc', DateTime(timezone=True), nullable=False), # Timestamp from Reddit
    Column('fetched_at', DateTime(timezone=True), server_default=func.now()),
    # Indexes
    Index('ix_reddit_data_subreddit_created', 'subreddit', 'created_utc'),
    Index('ix_reddit_data_created_utc', 'created_utc')
)

twitter_data = Table(
    'twitter_data', metadata,
    Column('id', Integer, primary_key=True),
    Column('tweet_id', String(30), nullable=False, unique=True), # Twitter Tweet ID
    Column('author_id', String(30), nullable=True),
    Column('text', Text, nullable=False),
    Column('created_at', DateTime(timezone=True), nullable=False), # Timestamp from Twitter (UTC)
    Column('public_metrics', JSON_TYPE, nullable=True), # Store dict: {'retweet_count': .., 'reply_count': .., 'like_count': .., 'quote_count': ..}
    Column('hashtags', JSON_TYPE, nullable=True), # Store as list
    Column('cashtags', JSON_TYPE, nullable=True), # Store as list
    Column('fetched_at', DateTime(timezone=True), server_default=func.now()),
    # Indexes
    Index('ix_twitter_data_created_at', 'created_at')
)

sentiment_analysis_results = Table(
    'sentiment_analysis_results', metadata,
    Column('id', Integer, primary_key=True),
    # Foreign keys to link to the source data
    Column('news_id', Integer, nullable=True), # FK to news_data.id
    Column('reddit_id', Integer, nullable=True), # FK to reddit_data.id
    Column('twitter_id', Integer, nullable=True), # FK to twitter_data.id
    Column('model_name', String(100), nullable=False), # Sentiment model used
    Column('sentiment_label', String(20), nullable=False), # e.g., 'positive', 'negative', 'neutral'
    Column('sentiment_score', Float, nullable=False), # e.g., -1.0 to 1.0
    Column('analyzed_at', DateTime(timezone=True), server_default=func.now()),
    # Indexes - Index individual FKs or create composite indexes if needed
    Index('ix_sentiment_news_id', 'news_id'),
    Index('ix_sentiment_reddit_id', 'reddit_id'),
    Index('ix_sentiment_twitter_id', 'twitter_id'),
    Index('ix_sentiment_analyzed_at', 'analyzed_at')
    # TODO: Add constraint to ensure only one FK is non-null if DB supports it easily,
    # otherwise handle in application logic or triggers.
)

trade_log = Table(
    'trade_log', metadata,
    Column('id', Integer, primary_key=True),
    Column('timestamp', DateTime(timezone=True), server_default=func.now()),
    Column('symbol', String(32), nullable=False),
    Column('trade_type', String(4), nullable=False), # 'BUY' or 'SELL'
    Column('order_type', String(10), nullable=False), # 'MARKET', 'LIMIT', etc.
    Column('status', String(20), nullable=False), # 'FILLED', 'FAILED', 'SIMULATED'
    Column('binance_order_id', String(50), nullable=True), # Actual Order ID from Binance
    Column('price', Float, nullable=False), # Average execution price
    Column('quantity', Float, nullable=False),
    Column('usd_value', Float, nullable=False), # Total value of trade
    Column('fee', Float, nullable=True), # Trading fee paid
    Column('pnl', Float, nullable=True), # Realized Profit/Loss for SELL trades
    Column('signal_confidence', Float, nullable=True), # Confidence score from prediction model
    Column('trigger_reason', String(50), nullable=True), # 'Model Signal', 'Stop Loss', etc.
    Column('trading_mode', String(10), nullable=False), # 'PAPER' or 'LIVE'
    # Indexes
    Index('ix_trade_log_symbol_timestamp', 'symbol', 'timestamp'),
    Index('ix_trade_log_timestamp', 'timestamp')
)


# --- Database Utility Functions ---

def init_db():
    """Creates database tables if they don't exist."""
    if not engine:
        log.error("Cannot initialize database, engine not configured.")
        return False
    if not metadata.tables:
         log.warning("No tables defined in metadata, skipping init_db.")
         return False
    try:
        log.info("Attempting to create database tables if they don't exist...")
        # Check existing tables
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        log.info(f"Existing tables: {existing_tables}")

        # Create tables
        metadata.create_all(bind=engine)
        log.info("Database tables checked/created successfully.")

        # Verify creation (optional)
        inspector = inspect(engine) # Re-inspect after creation
        new_tables = inspector.get_table_names()
        log.info(f"Tables after creation attempt: {new_tables}")
        return True
    except SQLAlchemyError as e:
        log.error(f"SQLAlchemyError initializing database: {e}", exc_info=True)
    except Exception as e:
        log.error(f"Unexpected error initializing database: {e}", exc_info=True)
    return False

@contextmanager
def get_db_session():
    """Provides a database session using a context manager pattern."""
    if not SessionLocal:
        log.error("SessionLocal not configured. Cannot get DB session.")
        yield None # Or raise an exception
        return

    db = SessionLocal()
    try:
        log.debug("DB Session opened.")
        yield db
    except SQLAlchemyError as e:
         log.error(f"Database session error: {e}", exc_info=True)
         db.rollback()
         raise # Re-raise the exception after rollback
    except Exception as e:
         log.error(f"Unexpected error in DB session context: {e}", exc_info=True)
         db.rollback()
         raise
    finally:
        log.debug("DB Session closed.")
        db.close()

def bulk_insert_data(data_list: list[dict], table: Table, unique_column: str = None):
    """
    Inserts a list of dictionaries into the specified table, optionally skipping duplicates
    based on a unique column (like URL or post_id).

    Args:
        data_list (list[dict]): List of data records (dictionaries).
        table (Table): SQLAlchemy Table object to insert into.
        unique_column (str, optional): Column name to check for existence before inserting.
                                        If None, attempts to insert all records.
    Returns:
        int: Number of rows successfully inserted.
    """
    if not engine:
        log.error(f"Cannot insert data into {table.name}, engine not configured.")
        return 0
    if not data_list:
        log.debug(f"No data provided to insert into {table.name}.")
        return 0

    inserted_count = 0
    skipped_count = 0
    records_to_insert = []

    try:
        with engine.connect() as connection:
            transaction = connection.begin()  # Explicitly manage the transaction
            try:
                if unique_column and hasattr(table.c, unique_column): # Check if column exists
                    # Check for existing records if unique_column is provided
                    existing_values = set()
                    unique_values_in_batch = {record.get(unique_column) for record in data_list if record.get(unique_column)}

                    if unique_values_in_batch:
                        select_stmt = table.select().where(getattr(table.c, unique_column).in_(unique_values_in_batch))
                        result = connection.execute(select_stmt)
                        # Fetch the actual values from the result proxy
                        existing_values = {row[getattr(table.c, unique_column)] for row in result}
                        log.debug(f"Found {len(existing_values)} existing records in {table.name} based on {unique_column}.")

                    for record in data_list:
                        if record.get(unique_column) in existing_values:
                            skipped_count += 1
                        else:
                            records_to_insert.append(record)
                else:
                    if unique_column:
                        log.warning(f"Unique column '{unique_column}' not found in table '{table.name}'. Inserting all records without checking.")
                    # Insert all if no unique check needed or column doesn't exist
                    records_to_insert = data_list

                if not records_to_insert:
                    log.info(f"No new records to insert into {table.name}. Skipped {skipped_count} potential duplicates.")
                    transaction.rollback() # Rollback if nothing to insert
                    return 0

                # Perform bulk insert
                log.info(f"Attempting to insert {len(records_to_insert)} new records into {table.name} (skipped {skipped_count})...")
                connection.execute(table.insert(), records_to_insert)
                inserted_count = len(records_to_insert)  # Assume success if no exception

                transaction.commit()  # Commit the transaction
                log.info(f"Successfully inserted {inserted_count} rows into {table.name}.")

            except Exception as e:
                transaction.rollback()  # Rollback the transaction on error
                log.error(f"Error during bulk insert transaction into {table.name}: {e}", exc_info=True)
                inserted_count = 0 # Reset count on error

    except IntegrityError as e:
        # This might catch unique constraint violations not caught by the pre-check
        log.error(f"IntegrityError inserting data into {table.name}: {e}. Potential duplicate key violation.", exc_info=True)
        inserted_count = 0
    except SQLAlchemyError as e:
        log.error(f"SQLAlchemyError connecting or inserting data into {table.name}: {e}", exc_info=True)
        inserted_count = 0
    except Exception as e:
        log.error(f"Unexpected error inserting data into {table.name}: {e}", exc_info=True)
        inserted_count = 0

    return inserted_count


def df_to_db(df: pd.DataFrame, table_name: str, if_exists: str = 'append', index: bool = False, **kwargs):
    """
    Writes a Pandas DataFrame to the specified database table using SQLAlchemy engine.

    Args:
        df (pd.DataFrame): DataFrame to write.
        table_name (str): Name of the database table.
        if_exists (str): How to behave if the table already exists.
                         Options: 'fail', 'replace', 'append'. Default is 'append'.
        index (bool): Write DataFrame index as a column. Default is False.
        **kwargs: Additional arguments passed to pandas.DataFrame.to_sql().
    """
    if not engine:
        log.error(f"Cannot write DataFrame to {table_name}, engine not configured.")
        return
    if df.empty:
        log.info(f"DataFrame for table {table_name} is empty, skipping database write.")
        return

    log.info(f"Writing {len(df)} rows to table '{table_name}' (if_exists='{if_exists}')...")
    try:
        # Ensure column names match DB schema if appending (case sensitivity might matter)
        # Consider lowercasing df columns if DB schema uses lowercase
        # df.columns = df.columns.str.lower()

        df.to_sql(
            name=table_name,
            con=engine,
            if_exists=if_exists,
            index=index,
            chunksize=1000, # Write in chunks for better memory usage
            method='multi', # Use multi-value inserts if supported
            **kwargs
        )
        log.info(f"Successfully wrote DataFrame to table '{table_name}'.")
    except SQLAlchemyError as e:
        log.error(f"SQLAlchemyError writing DataFrame to {table_name}: {e}", exc_info=True)
    except Exception as e:
        log.error(f"Unexpected error writing DataFrame to {table_name}: {e}", exc_info=True)


def fetch_table_data(table_name: str) -> pd.DataFrame: # Add return type hint
    """
    Fetches all data from a specified table.

    Args:
        table_name (str): Name of the table to fetch data from.

    Returns:
        pd.DataFrame: DataFrame containing the table data. Empty DataFrame on error.
    """
    if not engine:
        log.error("Cannot fetch data, engine not configured.")
        return pd.DataFrame()

    try:
        with engine.connect() as connection:
            # Use SQLAlchemy Table object if available for safer quoting
            if table_name in metadata.tables:
                table_obj = metadata.tables[table_name]
                query = table_obj.select()
                result = pd.read_sql(query, connection)
            else:
                # Fallback to string query (less safe against SQL injection if table_name is dynamic)
                log.warning(f"Table '{table_name}' not found in metadata, using raw SQL query.")
                query = f"SELECT * FROM {table_name}" # Ensure table_name is validated if dynamic
                result = pd.read_sql(query, connection)

            log.info(f"Fetched {len(result)} rows from table '{table_name}'.")
            return result
    except SQLAlchemyError as e:
        log.error(f"SQLAlchemyError fetching data from {table_name}: {e}", exc_info=True)
    except Exception as e:
        log.error(f"Unexpected error fetching data from {table_name}: {e}", exc_info=True)

    return pd.DataFrame()


# Example of running init_db directly (for setup)
# Run using 'python -m hydrobot.database.db_utils' from the project root directory
if __name__ == '__main__':
    # Setup basic logging using the centralized setup function
    from hydrobot.utils.logger_setup import setup_logging
    import logging # Import standard logging only for fallback in __main__
    try:
        # Use log level from settings if available
        log_level_setting = getattr(settings, 'log_level', 'INFO')
        setup_logging(log_level_setting)
        log.info("Logging setup for __main__ test using centralized setup.")
    except Exception as e:
        print(f"Error setting up logging from settings: {e}")
        # Fallback basic config if setup fails
        logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s [%(name)s] %(message)s')
        log = logging.getLogger(__name__) # Re-assign log to fallback logger

    print("Attempting to initialize database schema (Direct Run)...")
    if DATABASE_URL: # Check if DATABASE_URL was successfully loaded
        if init_db():
             print("Database initialization check complete.")
        else:
             print("Database initialization failed. Check logs.")
    else:
         print("DATABASE_URL not found in settings.database.url, skipping initialization.")


