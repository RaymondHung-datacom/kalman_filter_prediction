import os
import re
import json
import uuid
from typing import List, Dict, Any
from datetime import datetime, date, timedelta

from sentence_transformers import SentenceTransformer, util
from qdrant_client import QdrantClient, models

# --- Configuration ---
REPORTS_DIR = "news_reports"

# This is a powerful multilingual model that works well for Japanese
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

# This query defines what "relevant" means. All scraped articles will be compared to this.
RELEVANCE_QUERY = "小売業界の動向、新製品、販売戦略、消費者トレンド"
# Adjust this threshold. Higher values mean stricter relevance checking. (0.0 to 1.0)
SIMILARITY_THRESHOLD = 0.3
# NEW: Minimum character count for content to be considered for vectorization.
MIN_CONTENT_LENGTH = 150
# NEW: Threshold for considering an article a duplicate. 0.98 means very high similarity.
DEDUPLICATION_THRESHOLD = 0.98

# Qdrant Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "retail_news_jp"

# def parse_publish_date(date_str: str) -> date:
#     """
#     Robust Japanese date parser. 
#     Handles: "2025/10/31", "10/31(金)", "2025-10-31T...", etc.
#     """
#     if not date_str:
#         return None
    
#     current_year = datetime.now().year
    
#     try:
#         # 1. ISO Format (2025-10-31T...)
#         if 'T' in date_str:
#             return datetime.fromisoformat(date_str).date()
            
#         # 2. Yahoo/Japanese formats (10/31(金) or 2023/10/31)
#         # Remove Japanese day of week: (金) -> empty
#         clean_str = re.sub(r'\([土日月火水木金]\)', '', date_str).strip()
        
#         # Split by slash or hyphen
#         parts = re.split(r'[-/]', clean_str)
        
#         if len(parts) == 3: # Year, Month, Day
#             return date(int(parts[0]), int(parts[1]), int(parts[2]))
#         elif len(parts) == 2: # Month, Day (Assume current year)
#             return date(current_year, int(parts[0]), int(parts[1]))
            
#     except Exception as e:
#         print(f"⚠️ Date parse error: {date_str}")
#         return None
#     return None

def load_articles_from_reports(directory: str) -> List[Dict[str, Any]]:
    """Loads all articles from all JSON files in a directory."""
    all_articles = []
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found. Please run news_crawler.py first.")
        return []

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                    print(f"Loaded {len(articles)} articles from {filename}")
                    all_articles.extend(articles)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Could not read or parse {filename}: {e}")
    return all_articles


def is_article_from_today(article: Dict[str, Any], current_year: int) -> bool:
    """
    Checks if an article was published today. Handles multiple date formats.
    """
    publish_date_str = article.get("publish_date")
    if not publish_date_str:
        return False

    today = date.today()
    article_date = None

    try:
        # Format 1: "10/31(金) 11:16" (from Yahoo News)
        if '(' in publish_date_str and ')' in publish_date_str:
            # The year is missing, so we use the current year.
            # This assumes crawls happen for the current year's news.
            date_part = publish_date_str.split('(')[0]
            parsed_date = datetime.strptime(f"{current_year}/{date_part}", "%Y/%m/%d").date()
            article_date = parsed_date

        # Format 2: "2025-10-31T11:16:00+09:00" (from Livedoor News)
        elif 'T' in publish_date_str:
            article_date = datetime.fromisoformat(publish_date_str).date()

    except (ValueError, TypeError):
        # If parsing fails, skip the article
        print(f"  - Warning: Could not parse date '{publish_date_str}' for article '{article.get('title', 'N/A')[:30]}...'. Skipping.")
        return False

    return article_date == today

def title_exists(client: QdrantClient, collection_name: str, title: str) -> bool:
    """Checks if an article with the exact title already exists in the collection."""
    response = client.scroll(collection_name=collection_name, scroll_filter=models.Filter(must=[models.FieldCondition(key="title", match=models.MatchValue(value=title))]), limit=1)
    return len(response[0]) > 0

if __name__ == "__main__":
    # 1. Initialize the embedding model
    print(f"Loading sentence transformer model: '{MODEL_NAME}'...")
    # This may take a few minutes on the first run as it downloads the model
    model = SentenceTransformer(MODEL_NAME)
    vector_size = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Vector size: {vector_size}")

    print("Vectorizing master relevance query...")
    master_relevance_vector = model.encode(RELEVANCE_QUERY)

    # Add a check to ensure vector_size is an integer
    if not isinstance(vector_size, int):
        print("Error: Could not determine the vector size from the model.")
        exit()

    # 2. Initialize Qdrant client and create collection
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")

    # Check if collection exists, if not, create it
    try:
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    except Exception: # QdrantClient raises ValueError if collection not found
        print(f"Collection '{COLLECTION_NAME}' not found. Creating it...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        print("Collection created successfully.")

    # 3. Load articles
    articles_to_process = load_articles_from_reports(REPORTS_DIR)
    if not articles_to_process:
        print("No articles to process. Exiting.")
        exit()

    # 4. Generate embeddings and prepare for upload
    points_to_upload = []
    relevant_count = 0
    today_count = 0
    current_year = datetime.now().year

    for article in articles_to_process:
        title = article.get("title", "")
        content = article.get("content", "")

        # --- NEW: Filter for articles published today ---
        if not is_article_from_today(article, current_year):
            continue
        today_count += 1

        # --- NEW: Check if a document with the exact same title already exists ---
        if title_exists(client, COLLECTION_NAME, title):
            print(f"Skipping '{title[:30]}...' as an article with the same title already exists.")
            continue

        # NEW: Filter out articles with content that is too short to be meaningful.
        if len(content) < MIN_CONTENT_LENGTH:
            print(f"Skipping '{title[:30]}...' due to short content (length: {len(content)}).")
            continue

        # Combine title and content for a richer context, especially for short articles.
        text_to_vectorize = f"{title}\n{content}"

        if text_to_vectorize.strip():  # Ensure we have some text to process
            article_vector = model.encode(text_to_vectorize)

            # --- 1. Deduplication Check ---
            # Search for vectors that are extremely similar to the current one.
            hits = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=article_vector.tolist(),
                limit=1,
                score_threshold=DEDUPLICATION_THRESHOLD,
            )

            if hits:
                print(f"Skipping '{title[:30]}...' as a near-duplicate was found.")
                continue

            # --- 2. Relevance Check ---
            similarity = util.cos_sim(master_relevance_vector, article_vector)[0][0].item()

            print(f"Processing '{title[:30]}...' | Unique | Relevance: {similarity:.4f}")

            if similarity >= SIMILARITY_THRESHOLD:
                print("  -> Relevant! Adding to upload queue.")
                relevant_count += 1
                # Prepare the data payload to store alongside the vector
                payload = {
                    "title": article.get("title"),
                    "link": article.get("link"),
                    "publish_date": article.get("publish_date"), # Add publish_date
                    "scraped_at": article.get("scraped_at"),
                    "content": content  # Store the full content
                }
                
                # Create a Qdrant Point
                points_to_upload.append(
                    models.PointStruct(id=str(uuid.uuid4()), vector=article_vector.tolist(), payload=payload)
                )
            else:
                print("  -> Not relevant. Skipping.")

    # 5. Upload the data to Qdrant in batches
    if points_to_upload:
        print(f"\nProcessed {today_count} articles from today.")
        print(f"Found {relevant_count} new, relevant articles to add.")
        print(f"Uploading {len(points_to_upload)} points to Qdrant...")
        client.upsert(collection_name=COLLECTION_NAME, points=points_to_upload, wait=True)
        print("Upload complete!")
    else:
        print(f"\nProcessed {today_count} articles from today, but no new, relevant articles were found to upload.")