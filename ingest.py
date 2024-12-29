import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Set
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
import time
from dotenv import load_dotenv
import json
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Pinecone index configuration
index_name = "insurance-articles"
dimension = 1536  # OpenAI embedding dimension

# Create index if it doesn't exist
try:
    index = pc.Index(index_name)
    print(f"Connected to existing index: {index_name}")
except Exception as e:
    print(f"Index {index_name} not found, creating...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=pinecone.ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Created new index: {index_name}")
    # Wait for index to be ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
    index = pc.Index(index_name)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Enable to not upload to Pinecone
MOCK_UPLOAD = False

# Source configurations
SOURCES = {
    "allstate": {
        "base_urls": ["https://www.allstate.com/resources/car-insurance"],
        "article_patterns": ["/resources/car-insurance/"],
        "excluded_terms": ["quote", "bundle", "calculator", "español", "moving", "disaster", "flood"],
        "content_selectors": ["#main-content [class*='content']", "#main-content [class*='article']"],
        "max_articles": 20
    },
    "progressive": {
        "base_urls": [
            "https://www.progressive.com/answers/",
            "https://www.progressive.com/answers/car-insurance/"
        ],
        "article_patterns": ["/answers/", "/learn/"],
        "excluded_terms": ["quote", "bundle", "calculator", "español"],
        "content_selectors": [".main-content", ".article-content"],
        "max_articles": 20
    },
    "auto-owners": {
        "base_urls": [
            "https://www.auto-owners.com/insurance",
            "https://www.auto-owners.com/insurance/auto"
        ],
        "article_patterns": ["/insurance/", "/resources/"],
        "excluded_terms": ["quote", "find-agent", "claims/report"],
        "content_selectors": [".content-area", ".article-content"],
        "max_articles": 20
    }
}

def setup_driver():
    """Setup and return a configured Firefox WebDriver."""
    firefox_options = Options()
    firefox_options.add_argument("--headless")
    firefox_options.add_argument("--disable-gpu")
    firefox_options.add_argument("--window-size=1920,1080")
    firefox_options.add_argument("--start-maximized")
    firefox_options.add_argument("--disable-blink-features=AutomationControlled")
    firefox_options.set_preference("dom.webdriver.enabled", False)
    firefox_options.set_preference('useAutomationExtension', False)
    
    # Add common user agent
    firefox_options.set_preference('general.useragent.override', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = webdriver.Firefox(options=firefox_options)
    driver.set_page_load_timeout(30)
    return driver

def wait_for_page_load(driver, url):
    """Wait for the page to fully load and render content."""
    print(f"Loading {url}...")
    driver.get(url)
    
    # Wait for initial page load
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    
    print("Initial page load complete, waiting for content...")
    
    # Wait for potential overlay/loading elements to disappear
    try:
        WebDriverWait(driver, 10).until_not(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[class*='loading'], [class*='overlay']"))
        )
    except:
        print("No loading overlay found or timeout waiting for it to disappear")
    
    # Additional wait for dynamic content
    time.sleep(5)
    
    # Scroll to bottom to trigger lazy loading
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    
    # Scroll back to top
    driver.execute_script("window.scrollTo(0, 0);")
    
    print("Page fully loaded")

def get_article_urls(source_config: dict, base_url: str) -> set:
    """Collect article URLs from a base page based on source configuration."""
    article_urls = set()
    try:
        driver = setup_driver()
        
        print(f"\n=== Fetching from {base_url} ===")
        wait_for_page_load(driver, base_url)
        
        # Get page title for debugging
        print(f"\nPage Title: {driver.title}\n")
        
        if "Access Denied" in driver.title or "Security Check" in driver.page_source:
            print("WARNING: Possible security check or access denied page")
            print("Page source preview:")
            print(driver.page_source[:500])
            return article_urls
        
        print("Looking for article links...")
        
        # Try multiple strategies to find content areas
        content_areas = []
        
        # Strategy 1: Main content area
        try:
            main_content = driver.find_element(By.TAG_NAME, "main")
            if main_content:
                print("Found main content area")
                content_areas.append(main_content)
        except Exception as e:
            print(f"No main content area found: {str(e)}")
        
        # Strategy 2: Article containers by role
        try:
            article_elements = driver.find_elements(By.CSS_SELECTOR, "[role='article']")
            if article_elements:
                print(f"Found {len(article_elements)} article elements by role")
                content_areas.extend(article_elements)
        except Exception as e:
            print(f"No article elements found by role: {str(e)}")
        
        # Strategy 3: Source-specific content selectors
        for selector in source_config.get("content_selectors", []):
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    print(f"Found {len(elements)} elements using selector: {selector}")
                    content_areas.extend(elements)
            except Exception as e:
                continue
        
        if not content_areas:
            print("No content areas found with primary strategies, falling back to page scan")
            content_areas = [driver.find_element(By.TAG_NAME, "body")]
        
        print(f"\nTotal content areas to search: {len(content_areas)}")
        
        # Process each content area
        for area_idx, content_area in enumerate(content_areas, 1):
            print(f"\nSearching content area {area_idx}/{len(content_areas)}")
            
            # Look for any links
            try:
                links = content_area.find_elements(By.TAG_NAME, "a")
                if links:
                    print(f"Found {len(links)} potential links")
                    
                    for link in links:
                        if len(article_urls) >= source_config["max_articles"]:
                            print(f"\nReached maximum article limit ({source_config['max_articles']})")
                            return article_urls
                        
                        href = link.get_attribute("href")
                        text = link.text.strip()
                        
                        if not href or not text:
                            continue
                        
                        # Skip non-http links
                        if not href.startswith("http"):
                            continue
                        
                        # Ensure link is from the same domain
                        if not urlparse(href).netloc == urlparse(base_url).netloc:
                            continue
                        
                        print(f"\nEvaluating link:")
                        print(f"Text: {text}")
                        print(f"URL: {href}")
                        
                        # Check if URL matches any of the article patterns
                        if not any(pattern in href for pattern in source_config["article_patterns"]):
                            print("Skipping: Does not match article patterns")
                            continue
                        
                        # Check for excluded terms
                        if any(term in href.lower() for term in source_config["excluded_terms"]):
                            print(f"Skipping: Contains excluded term")
                            continue
                        
                        if href in article_urls:
                            print("Skipping: Duplicate article")
                            continue
                        
                        print("✓ Adding article to collection")
                        article_urls.add(href)
                        
            except Exception as e:
                print(f"Error processing links: {str(e)}")
                continue
        
        print(f"\nFound {len(article_urls)} relevant articles")
        
    except Exception as e:
        print(f"Error fetching article URLs: {str(e)}")
    finally:
        driver.quit()
    
    return article_urls

def scrape_article(url: str, source_config: dict) -> Dict:
    """Scrape a single article page using Selenium to handle dynamic content."""
    print(f"\n=== Processing article: {url} ===")
    
    try:
        print("Setting up browser...")
        driver = setup_driver()
        wait_for_page_load(driver, url)
        
        # Wait for content to load
        print("Waiting for content to load...")
        
        # Try source-specific content selectors first
        content = ""
        for selector in source_config["content_selectors"]:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    content += "\n".join(element.text for element in elements if element.text.strip())
            except Exception:
                continue
        
        # If no content found, try common fallback selectors
        if not content:
            fallback_selectors = ["article", "main", ".content", "#content"]
            for selector in fallback_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        content += "\n".join(element.text for element in elements if element.text.strip())
                except Exception:
                    continue
        
        if content:
            print(f"Found content length: {len(content)} characters")
            
            return {
                "url": url,
                "title": driver.title,
                "content": content
            }
        else:
            print("No content found")
            
    except Exception as e:
        print(f"Error scraping article: {str(e)}")
    finally:
        driver.quit()
    
    return None

def process_and_upload_articles(article_urls: Set[str], source_config: dict):
    """Process articles and upload them to Pinecone."""
    print("\nInitializing Pinecone connection...")
    try:
        index = pc.Index(index_name)
        print("Successfully connected to Pinecone index")
    except Exception as e:
        print(f"Error connecting to Pinecone: {str(e)}")
        return
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    batch = []
    for url in article_urls:
        article = scrape_article(url, source_config)
        if article and article["content"]:
            print(f"\nProcessing article: {article['title']}")
            
            # Split content into chunks
            chunks = text_splitter.split_text(article["content"])
            print(f"Split into {len(chunks)} chunks")
            
            # Create embeddings for each chunk
            for i, chunk in enumerate(chunks):
                try:
                    print(f"Creating embedding for chunk {i+1}/{len(chunks)}")
                    embedding = embeddings.embed_query(chunk)
                    
                    batch.append({
                        "id": f"{url}-{i}",
                        "values": embedding,
                        "metadata": {
                            "url": url,
                            "title": article["title"],
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "text": chunk,
                            "source": urlparse(url).netloc
                        }
                    })
                    
                except Exception as e:
                    print(f"Error creating embedding: {str(e)}")
                    continue
    
    if batch and not MOCK_UPLOAD:
        try:
            print(f"\nUploading {len(batch)} vectors to Pinecone...")
            index.upsert(vectors=batch)
            print("Successfully uploaded vectors")
        except Exception as e:
            print(f"Error uploading to Pinecone: {str(e)}")
    else:
        print("\nMock mode: No vectors uploaded" if MOCK_UPLOAD else "No vectors to upload - no content was successfully processed")

def main():
    """Main function to process all sources."""
    all_articles = set()
    
    for source_name, config in SOURCES.items():
        print(f"\n=== Processing {source_name.upper()} ===")
        for base_url in config["base_urls"]:
            articles = get_article_urls(config, base_url)
            if articles:
                all_articles.update(articles)
                process_and_upload_articles(articles, config)
    
    print(f"\nTotal unique article URLs collected: {len(all_articles)}")

if __name__ == "__main__":
    main()