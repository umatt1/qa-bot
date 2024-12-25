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

def setup_driver():
    """Setup and return a configured Firefox WebDriver."""
    firefox_options = Options()
    firefox_options.add_argument("--headless")
    firefox_options.add_argument("--disable-gpu")
    
    # Use local Firefox installation instead of downloading
    driver = webdriver.Firefox(options=firefox_options)
    return driver

def get_article_urls(base_url: str, max_articles: int = 5) -> set:
    """Collect article URLs from a base page, limited to max_articles."""
    article_urls = set()
    try:
        driver = setup_driver()
        print(f"\n=== Fetching from {base_url} ===")
        driver.get(base_url)
        print("Page loaded, waiting for content...")
        
        # Get page title for debugging
        print(f"\nPage Title: {driver.title}\n")
        
        # Look specifically for the article container
        print("Looking for article links...")
        
        # Try to find links within main content area first
        main_content = driver.find_element(By.TAG_NAME, "main")
        if main_content:
            # Look for article cards or content sections
            article_sections = main_content.find_elements(By.CSS_SELECTOR, 
                "div[class*='article'], div[class*='content'], div[class*='resource']")
            
            if not article_sections:
                # Fallback to looking for lists that might contain articles
                article_sections = main_content.find_elements(By.CSS_SELECTOR, "ul li")
            
            for section in article_sections:
                if len(article_urls) >= max_articles:
                    break
                    
                try:
                    # Try to find link within the section
                    link = section.find_element(By.TAG_NAME, "a")
                    href = link.get_attribute("href")
                    text = link.text.strip()
                    
                    # Only include actual article pages
                    if (href 
                        and text 
                        and "/resources/car-insurance/" in href 
                        and not any(x in href.lower() for x in [
                            "quote", "bundle", "calculator", "resources/car-insurance$",
                            "español", "moving", "disaster", "flood"
                        ])
                        and text.lower() not in ["auto", "car insurance", "resources"]
                    ):
                        print(f"\nPotential article found:")
                        print(f"Title: {text}")
                        print(f"URL: {href}")
                        article_urls.add(href)
                        
                except Exception as e:
                    continue
                    
        print(f"\nFound {len(article_urls)} relevant articles")
                
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
                    batch.append({
                        "id": f"{url}-{i}",
                        "values": embedding,
                        "metadata": {
                            "url": url,
                            "title": article["title"],
                            "content": chunk
                        }
                    })
                except Exception as e:
                    print(f"Error creating embedding: {str(e)}")
                    continue
    
    # Upload batch to Pinecone
    if batch:
        try:
            print(f"\nUploading {len(batch)} vectors to Pinecone...")
            index.upsert(vectors=batch)
            print("Successfully uploaded to Pinecone!")
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