import requests
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

# --- Scraper Functions ---

def get_daily_co_jp_content(soup: BeautifulSoup) -> Optional[str]:
    """Scrapes the main article text from a daily.co.jp article."""
    article_body = soup.find('p', class_='article-body')
    if article_body:
        return ' '.join(article_body.stripped_strings)
    return None

def get_generic_content(soup: BeautifulSoup) -> Optional[str]:
    """A generic fallback scraper that looks for common HTML5 tags like <article>."""
    article_tag = soup.find('article')
    if article_tag:
        return ' '.join(article_tag.stripped_strings)
    return None

def get_full_content_from_url(url: str) -> Optional[str]:
    """
    Scrapes the main article text from a given URL, handling Yahoo "pickup" pages.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Handle Yahoo "pickup" pages that link to external sites
        more_link_tag = soup.find('a', attrs={'data-ual-gotocontent': 'true'})
        external_url = more_link_tag.get('href') if more_link_tag else None

        if isinstance(external_url, str):
            print(f"    -> Pickup page detected. Following link to: {external_url}")
            time.sleep(1) # Be respectful to the server
            ext_response = requests.get(external_url, headers=headers, timeout=10)
            ext_response.raise_for_status()
            ext_soup = BeautifulSoup(ext_response.content, 'html.parser')

            if "daily.co.jp" in external_url:
                return get_daily_co_jp_content(ext_soup)
            else:
                print(f"    -> No specific scraper for {external_url}, using generic fallback.")
                return get_generic_content(ext_soup)
        else:
            # Original logic for full articles on Yahoo's site
            article_body = soup.find('div', class_='article_body')
            if not article_body:
                article_body = soup.find('p', class_='sc-iMCRTP')

            if article_body:
                return ' '.join(article_body.stripped_strings)
            else:
                print(f"  - Warning: Could not find standard article body for {url}")
                return None
    except requests.exceptions.RequestException as e:
        print(f"  - Error fetching {url}: {e}")
    return None


def search_and_scrape_yahoo_news(query: str, debug_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Searches Yahoo News for a query, scrapes the results, and returns the full content.

    Args:
        query: The topic to search for.
        debug_file: Optional path to a local HTML file for debugging.

    Returns:
        A list of article dictionaries containing full content.
    """
    search_url = "https://news.yahoo.co.jp/search"
    params = {'p': query}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        if debug_file and os.path.exists(debug_file):
            print(f"--- DEBUG MODE: Reading from local file: {debug_file} ---")
            with open(debug_file, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
        else:
            print(f"Searching Yahoo News for '{query}'...")
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

        results_list = soup.find('ol', class_='newsFeed_list')
        if not results_list:
            print("Error: Could not find the main list of search results ('ol.newsFeed_list').")
            return []
        
        search_results = results_list.find_all('li', recursive=False)
        print(f"Found {len(search_results)} potential articles.")

        scraped_articles = []
        for item in search_results:
            link_tag = item.find('a')
            if not link_tag:
                continue
            
            title_tag = link_tag.find('div', class_=lambda c: isinstance(c, str) and 'dHAJpi' in c)
            title = title_tag.text.strip() if title_tag else "No Title Found"
            
            time_tag = item.find('time')
            publish_date = time_tag.text if time_tag else "No Date Found"
            url = link_tag.get('href')

            if not isinstance(url, str):
                continue

            if "/articles/" not in url and "/pickup/" not in url:
                continue

            print(f"\n  - Scraping content for: {title[:40]}...")
            
            if not debug_file:
                time.sleep(1) 
            
            full_content = get_full_content_from_url(url)

            if full_content:
                scraped_articles.append({
                    "title": title,
                    "link": url,
                    "content": full_content,
                    "publish_date": publish_date,
                    "source_id": "Yahoo News",
                    "scraped_at": datetime.now().isoformat()
                })
                print(f"    -> Success! Scraped {len(full_content)} characters.")
            else:
                print(f"    -> Failed to scrape content.")
        
        return scraped_articles

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while searching Yahoo News: {e}")
        return []


def save_full_report(query: str, articles: List[Dict[str, Any]]):
    """Saves the fully scraped articles to a timestamped JSON file."""
    reports_dir = "news_reports"
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"yahoo_{query.lower()}_{timestamp}.json"
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
    DEBUG_FILE = os.path.join(SCRIPT_DIR, "yahoo_search_result.html")

    QUERIES_TO_SEARCH = load_queries("queries.txt")
    if not QUERIES_TO_SEARCH:
        print("No queries to process. Exiting.")
        exit()

    queries = ["新製品"] if DEBUG_MODE else QUERIES_TO_SEARCH

    for query in queries:
        print(f"\n{'='*20} Processing query: '{query}' for Yahoo News {'='*20}")
        
        full_articles = search_and_scrape_yahoo_news(query, debug_file=DEBUG_FILE if DEBUG_MODE else None)

        if full_articles:
            save_full_report(query, full_articles)
        else:
            print(f"No articles with full content could be scraped for '{query}'.")

        print(f"\n{'='*62}")
    
    print(f"\n--- YAHOO NEWS CRAWL COMPLETE ---")
