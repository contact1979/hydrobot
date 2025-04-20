# /sentiment_analysis/analyzer.py

import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Optional, Tuple
import datetime
import pytz
import time

# Use absolute imports assuming 'cyclonev2' is the project root added to PYTHONPATH
import config
# from .. import config # Use relative import if running as part of a package

log = logging.getLogger(__name__)

# --- Model Initialization ---
_sentiment_pipeline = None
_tokenizer = None
_model = None

def load_sentiment_model() -> Optional[Tuple[pipeline, AutoTokenizer]]:
    """
    Loads the Hugging Face sentiment analysis pipeline specified in the config.
    Uses a singleton pattern to load the model only once.

    Returns:
        Optional[Tuple[pipeline, AutoTokenizer]]: The loaded pipeline and tokenizer,
                                                  or None if loading fails.
    """
    global _sentiment_pipeline, _tokenizer, _model
    if _sentiment_pipeline is None:
        model_name = config.SENTIMENT_MODEL_NAME
        log.info(f"Attempting to load sentiment analysis model: {model_name}")
        start_time = time.time()
        try:
            # Load tokenizer and model explicitly for more control if needed
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Explicitly load model if pipeline doesn't handle device placement well later
            # _model = AutoModelForSequenceClassification.from_pretrained(model_name)
            # device = 0 if torch.cuda.is_available() else -1 # Use GPU if available

            _sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name, # Can pass loaded _model here if needed
                tokenizer=_tokenizer,
                # device=device, # Specify device if needed
                # Consider adding truncation=True here if it applies globally
            )
            end_time = time.time()
            log.info(f"Sentiment analysis model '{model_name}' loaded successfully in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            log.error(f"Error loading sentiment model '{model_name}': {e}", exc_info=True)
            _sentiment_pipeline = None
            _tokenizer = None
            _model = None

    if _sentiment_pipeline and _tokenizer:
         return _sentiment_pipeline, _tokenizer
    else:
         return None


def analyze_sentiment_batch(items_with_text: List[Dict], text_key: str = 'text') -> List[Dict]:
    """
    Analyzes sentiment for text within a list of dictionaries using the loaded pipeline.

    Args:
        items_with_text (List[Dict]): A list of dictionaries, where each dictionary
                                      must contain at least the key specified by `text_key`.
                                      It should also contain identifier keys (like 'id', 'source_type')
                                      to link the results back.
        text_key (str): The key in each dictionary that holds the text to be analyzed.
                        Defaults to 'text'.

    Returns:
        List[Dict]: A list of dictionaries, each containing:
                    - 'source_id': The original 'id' from the input item.
                    - 'sentiment_label': 'positive', 'negative', or 'neutral' (lowercase).
                    - 'sentiment_score': Float score (-1.0 to 1.0, positive for positive, negative for negative).
                    - 'analyzed_at': Timestamp of analysis (timezone-aware UTC datetime).
                    Returns an empty list if analysis fails or no valid text is found.
    """
    model_info = load_sentiment_model()
    if not model_info:
        log.error("Sentiment pipeline not available. Cannot analyze sentiment.")
        return []

    pipeline, tokenizer = model_info

    if not items_with_text:
        log.warning("No items provided for sentiment analysis.")
        return []

    results = []
    texts_to_analyze = []
    original_indices = [] # To map results back to original items

    # Prepare batch of texts and keep track of original item IDs
    log.info(f"Preparing {len(items_with_text)} items for sentiment analysis...")
    for i, item in enumerate(items_with_text):
        text = item.get(text_key)
        item_id = item.get('id') # Assuming 'id' is the primary key from the source table

        if item_id is None:
            log.warning(f"Skipping item at index {i} due to missing 'id'. Item keys: {item.keys()}")
            continue

        # Basic text cleaning (optional, can be expanded)
        if isinstance(text, str) and text.strip():
            cleaned_text = text.strip().replace('\n', ' ').replace('\r', '') # Remove newlines for analysis
            texts_to_analyze.append(cleaned_text)
            original_indices.append(i)
        else:
            log.debug(f"Skipping item id={item_id} due to missing or invalid text.")

    if not texts_to_analyze:
        log.warning("No valid text found in items for sentiment analysis.")
        return []

    log.info(f"Analyzing sentiment for {len(texts_to_analyze)} text items...")
    analysis_timestamp = datetime.datetime.now(pytz.utc)
    analysis_results = []

    try:
        # Process in batches if the list is very large (pipeline might handle this internally too)
        # batch_size = 64 # Example batch size
        # for i in range(0, len(texts_to_analyze), batch_size):
        #     batch_texts = texts_to_analyze[i:i + batch_size]
        #     batch_results = pipeline(batch_texts, truncation=True, max_length=tokenizer.model_max_length)
        #     analysis_results.extend(batch_results)

        # Process all at once (simpler for moderate numbers)
        analysis_results = pipeline(
            texts_to_analyze,
            truncation=True,            # Truncate texts longer than model max length
            max_length=tokenizer.model_max_length, # Use tokenizer's max length
            # padding=True, # Usually handled by pipeline
            # return_all_scores=False # Only return the top score/label
            )
        log.info("Sentiment analysis pipeline execution complete.")

        if len(analysis_results) != len(original_indices):
             log.error(f"Mismatch in sentiment output count ({len(analysis_results)}) vs input text count ({len(texts_to_analyze)}).")
             return [] # Avoid partial results or incorrect mapping

        # Process results and map back
        for i, sentiment_output in enumerate(analysis_results):
            original_item_index = original_indices[i]
            original_item = items_with_text[original_item_index]
            original_item_id = original_item['id'] # We ensured ID exists earlier

            label = sentiment_output['label'].lower()
            score = sentiment_output['score']
            normalized_score = 0.0

            # Normalize score: positive label -> positive score, negative label -> negative score
            # Adjust based on model's specific labels (e.g., 'POSITIVE'/'NEGATIVE', 'LABEL_1'/'LABEL_0')
            # FinBERT typically uses 'positive', 'negative', 'neutral'
            if label == 'positive':
                normalized_score = score
            elif label == 'negative':
                normalized_score = -score
            # Neutral score remains 0.0 or could be score if label is neutral (depends on desired scale)

            results.append({
                'source_id': original_item_id, # Link back to the original record ID
                'sentiment_label': label,
                'sentiment_score': round(normalized_score, 4),
                'analyzed_at': analysis_timestamp
            })

    except Exception as e:
        log.error(f"Error during sentiment analysis pipeline execution: {e}", exc_info=True)
        return [] # Return empty list on failure

    log.info(f"Successfully analyzed sentiment for {len(results)} items.")
    return results

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # Setup basic logging for direct script run
    logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s [%(name)s] %(message)s')

    print("--- Testing Sentiment Analyzer ---")

    # Ensure model is loaded (call function directly)
    load_sentiment_model()

    # Sample data mimicking fetched data (needs 'id' and a text key)
    sample_data = [
        {'id': 101, 'source_type': 'reddit', 'text': "This is great news, Bitcoin is going to the moon!🚀"},
        {'id': 102, 'source_type': 'twitter', 'text': "Lost all my savings in $XYZ coin, feeling devastated. Total scam."},
        {'id': 103, 'source_type': 'news', 'text': "Market analysts suggest a period of consolidation for major cryptocurrencies."},
        {'id': 104, 'source_type': 'reddit', 'text': ""}, # Empty text
        {'id': 105, 'source_type': 'twitter'}, # Missing text key
        {'id': 106, 'source_type': 'news', 'text': "Neutral outlook for the short term."},
    ]

    print(f"\nAnalyzing {len(sample_data)} sample items...")
    # Specify the key containing the text, e.g., 'text' or 'text_combined'
    sentiment_results = analyze_sentiment_batch(sample_data, text_key='text')

    if sentiment_results:
        print(f"\nAnalysis Results ({len(sentiment_results)}):")
        for res in sentiment_results:
            print(f"  Source ID: {res['source_id']}, Label: {res['sentiment_label']}, Score: {res['sentiment_score']:.4f}")
    else:
        print("\nSentiment analysis failed or produced no results. Check logs (model loading?).")

    print("\n--- Test Complete ---")

