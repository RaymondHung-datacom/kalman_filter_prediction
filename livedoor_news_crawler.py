import requests
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from urllib.parse import quote

# --- Scraper Functions ---

def get_generic_content(soup: BeautifulSoup) -> Optional[str]:
    """A generic fallback scraper that looks for common HTML5 tags like <article>."""
    article_tag = soup.find('article')
    if article_tag:
        # This is a simple approach; it might grab extra unwanted text.
        return ' '.join(p.text.strip() for p in article_tag.find_all('p'))
    return None


def get_livedoor_content(url: str) -> Optional[str]:
    """Scrapes the main article text from a Livedoor News URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        # Livedoor pages can be EUC-JP or UTF-8, so we let requests detect it
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')

        # --- DEFINITIVE FIX: Handle "topics" summary pages ---
        # These pages link to the full article within a div with class "articleMore"
        article_more_div = soup.find('div', class_='articleMore')
        read_more_link = article_more_div.find('a') if article_more_div else None
        
        full_article_url = read_more_link.get('href') if read_more_link else None

        if isinstance(full_article_url, str):
            print(f"    -> Summary page detected. Following link to full article...")
            
            time.sleep(1) # Be respectful to the server
            ext_response = requests.get(full_article_url, headers=headers, timeout=10)
            ext_response.raise_for_status()
            ext_response.encoding = ext_response.apparent_encoding
            ext_soup = BeautifulSoup(ext_response.text, 'html.parser')
            
            # Use a generic scraper for the external site
            return get_generic_content(ext_soup)

        # --- Original logic for full articles on Livedoor's site ---
        article_body = soup.find('div', class_='articleBody')
        
        # NEW: Add a fallback to check for the hyphenated class name
        if not article_body:
            article_body = soup.find('div', class_='article-body')
        
        # NEW: Add a third fallback for another common layout
        if not article_body:
            article_body = soup.find('div', class_='article-body-inner')

        if article_body:
            # Find all paragraphs within the article body
            paragraphs = article_body.find_all('p')
            full_text = ' '.join(p.text.strip() for p in paragraphs)
            return full_text
        else:
            print(f"  - Warning: Could not find a known article body structure for {url}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"  - Error fetching {url}: {e}")
    return None


def search_and_scrape_livedoor_news(query: str, debug_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Searches Livedoor News for a query, scrapes the results, and returns the full content.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        if debug_file and os.path.exists(debug_file):
            print(f"--- DEBUG MODE: Reading from local file: {debug_file} ---")
            with open(debug_file, 'r', encoding='euc-jp') as f:
                soup = BeautifulSoup(f, 'html.parser')
        else:
            # Encode the query using EUC-JP as discovered
            encoded_query = quote(query, encoding='euc-jp')
            search_url = f"https://news.livedoor.com/search/article/?ie=euc-jp&word={encoded_query}"
            
            print(f"Searching Livedoor News for '{query}'...")
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            # Livedoor pages are EUC-JP, so we need to decode them correctly
            response.encoding = 'euc-jp'
            soup = BeautifulSoup(response.text, 'html.parser')

        # The main container for search results is a 'ul' with class 'articleList'
        results_list = soup.find('ul', class_='articleList')
        if not results_list:
            print("Error: Could not find the main list of search results ('ul.articleList').")
            return []

        search_results = results_list.find_all('li')
        print(f"Found {len(search_results)} potential articles.")

        scraped_articles = []
        for item in search_results:
            link_tag = item.find('a')
            url = link_tag.get('href') if link_tag else None
            title_tag = item.find('h3', class_='articleListTtl')            
            time_tag = item.find('time')
            publish_date = time_tag['datetime'] if time_tag and 'datetime' in time_tag.attrs else "No Date Found"

            if title_tag and isinstance(url, str):
                title = title_tag.text.strip()
                print(f"\n  - Scraping content for: {title[:40]}...")
                
                if not debug_file:
                    time.sleep(1) # Be respectful to the server
                
                full_content = get_livedoor_content(url)

                if full_content:
                    scraped_articles.append({
                        "title": title,
                        "link": url,
                        "content": full_content,
                        "publish_date": publish_date,
                        "source_id": "Livedoor News",
                        "scraped_at": datetime.now().isoformat()
                    })
                    print(f"    -> Success! Scraped {len(full_content)} characters.")
                else:
                    print(f"    -> Failed to scrape content.")
        
        return scraped_articles

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while searching Livedoor News: {e}")
        return []


def save_full_report(query: str, articles: List[Dict[str, Any]]):
    """Saves the fully scraped articles to a timestamped JSON file."""
    reports_dir = "news_reports"
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"livedoor_{query.lower()}_{timestamp}.json"
    filepath = os.path.join(reports_dir, filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"\nSuccessfully saved {len(articles)} full articles to {filepath}")
    except IOError as e:
        print(f"Error saving file: {e}")


def load_queries(filepath: str) -> List[str]:
    """Loads search queries from a text file, one query per line."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        print(f"Loaded {len(queries)} queries from {filepath}")
        return queries
    except FileNotFoundError:
        print(f"Error: Query file not found at '{filepath}'. Please create it.")
        return []


if __name__ == "__main__":
    # --- Configuration ---
    DEBUG_MODE = False

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEBUG_FILE = os.path.join(SCRIPT_DIR, "livedoor_search_result.html")

    QUERIES_TO_SEARCH = load_queries("queries.txt")
    if not QUERIES_TO_SEARCH:
        print("No queries to process. Exiting.")
        exit()

    queries = ["新製品"] if DEBUG_MODE else QUERIES_TO_SEARCH

    for query in queries:
        print(f"\n{'='*20} Processing query: '{query}' for Livedoor News {'='*20}")
        
        full_articles = search_and_scrape_livedoor_news(query, debug_file=DEBUG_FILE if DEBUG_MODE else None)

        if full_articles:
            save_full_report(query, full_articles)
        else:
            print(f"No articles with full content could be scraped for '{query}'.")

        print(f"\n{'='*64}")
    
    print(f"\n--- LIVEDOOR NEWS CRAWL COMPLETE ---")