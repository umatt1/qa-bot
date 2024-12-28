import requests
from bs4 import BeautifulSoup
from typing import Dict, List
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
import time
from dotenv import load_dotenv
import json
from urllib.parse import urljoin
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

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Pinecone index name
index_name = "allstate-articles"

MOCK_UPLOAD = True

def setup_driver():
    """Setup and return a configured Firefox WebDriver."""
    firefox_options = Options()
    firefox_options.add_argument("--headless")
    firefox_options.add_argument("--disable-gpu")
    
    # Use local Firefox installation instead of downloading
    driver = webdriver.Firefox(options=firefox_options)
    return driver

def get_article_urls(base_url: str, max_articles: int = 10) -> set:
    """Collect article URLs from a base page, limited to max_articles."""
    article_urls = set()
    try:
        driver = setup_driver()
        print(f"\n=== Fetching from {base_url} ===")
        driver.get(base_url)
        print("Page loaded, waiting for content...")
        
        # Get page title for debugging
        print(f"\nPage Title: {driver.title}\n")
        
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
            
        # Strategy 3: Common content container classes
        content_selectors = [
            "div[class*='article-container']",
            "div[class*='content-container']",
            "div[class*='resource-list']",
            "section[class*='articles']",
            "div[class*='article-grid']"
        ]
        
        for selector in content_selectors:
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
            
            # Look for links with article-like content
            article_selectors = [
                "a[href*='/resources/']",
                "a[href*='/articles/']",
                "a[href*='/blog/']",
                "div[class*='article'] a",
                "div[class*='content'] a",
                "div[class*='resource'] a"
            ]
            
            for selector in article_selectors:
                try:
                    links = content_area.find_elements(By.CSS_SELECTOR, selector)
                    if links:
                        print(f"Found {len(links)} potential links using {selector}")
                        
                        for link in links:
                            if len(article_urls) >= max_articles:
                                print(f"\nReached maximum article limit ({max_articles})")
                                return article_urls
                                
                            href = link.get_attribute("href")
                            text = link.text.strip()
                            
                            if not href or not text:
                                continue
                                
                            print(f"\nEvaluating link:")
                            print(f"Text: {text}")
                            print(f"URL: {href}")
                            
                            # Refined filtering criteria
                            if not "/resources/car-insurance/" in href:
                                print("Skipping: Not in car insurance resources")
                                continue
                                
                            # Exclude utility pages but allow more article content
                            excluded_terms = [
                                "quote", "bundle", "calculator",
                                "español", "moving", "disaster", "flood"
                            ]
                            
                            if any(x in href.lower() for x in excluded_terms):
                                print(f"Skipping: Contains excluded term")
                                continue
                                
                            # More permissive text filtering
                            generic_terms = ["auto", "car insurance", "resources", "home"]
                            if text.lower() in generic_terms and len(text.split()) <= 2:
                                print("Skipping: Generic navigation link")
                                continue
                            
                            if href in article_urls:
                                print("Skipping: Duplicate article")
                                continue
                                
                            print("✓ Adding article to collection")
                            article_urls.add(href)
                            
                except Exception as e:
                    print(f"Error processing selector {selector}: {str(e)}")
                    continue
                    
        print(f"\nFound {len(article_urls)} relevant articles")
        if len(article_urls) == 0:
            print("\nDumping page source for debugging...")
            print(driver.page_source[:1000] + "...")
                
    except Exception as e:
        print(f"Error fetching article URLs: {str(e)}")
    finally:
        driver.quit()
        
    return article_urls

def scrape_article(url: str) -> Dict:
    """Scrape a single article page using Selenium to handle dynamic content."""
    print(f"\n=== Processing article: {url} ===")
    
    try:
        print("Setting up browser...")
        driver = setup_driver()
        driver.get(url)
        
        # Wait for content to load
        print("Waiting for content to load...")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "main-content"))
        )
        
        # Give extra time for React content to render
        time.sleep(2)
        
        # Get the page source after JavaScript has run
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Extract title
        title = driver.title
        print(f"Found title: {title}")
        
        # Look for content in the React app
        content_elements = driver.find_elements(By.CSS_SELECTOR, 
            "#main-content [class*='content'], #main-content [class*='article'], #main-content [class*='text']")
        
        if content_elements:
            # Combine text from all content elements
            text = "\n".join(element.text for element in content_elements if element.text.strip())
            print(f"Found content length: {len(text)} characters")
            
            if text.strip():
                return {
                    "url": url,
                    "title": title,
                    "content": text[:1000]  # Limit content for testing
                }
            else:
                print("Content elements found but no text extracted")
        else:
            print("No content elements found")
            
    except Exception as e:
        print(f"Error scraping article: {str(e)}")
    finally:
        driver.quit()
    
    return None

def process_and_upload_articles(article_urls: set):
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
        article = scrape_article(url)
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
                    
                    # Store the chunk text in the metadata and as page_content
                    batch.append({
                        "id": f"{url}-{i}",
                        "values": embedding,
                        "metadata": {
                            "url": url,
                            "title": article["title"],
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "text": chunk  # Add text to metadata
                        }
                    })
                    
                except Exception as e:
                    print(f"Error creating embedding: {str(e)}")
                    continue
    
    # Upload batch to Pinecone
    if batch:
        try:
            if not MOCK_UPLOAD:
                print(f"\nUploading {len(batch)} vectors to Pinecone...")
                index.upsert(vectors=batch)
                print("Successfully uploaded to Pinecone!")
            else:
                print("\nMock mode: No vectors uploaded")
        except Exception as e:
            print(f"Error uploading to Pinecone: {str(e)}")
    else:
        print("\nNo vectors to upload - no content was successfully processed")

if __name__ == "__main__":
    # Start with just car insurance articles
    base_url = "https://www.allstate.com/resources/car-insurance"
    
    # Collect limited number of article URLs
    article_urls = get_article_urls(base_url, max_articles=3)
    print(f"\nTotal unique article URLs collected: {len(article_urls)}")
    
    # Process and upload articles
    process_and_upload_articles(article_urls)