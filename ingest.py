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
        
        # Look for article links
        print("Looking for article links...")
        links = driver.find_elements(By.TAG_NAME, "a")
        print(f"Found {len(links)} total links\n")
        
        for link in links:
            if len(article_urls) >= max_articles:
                break
                
            try:
                href = link.get_attribute("href")
                text = link.text.strip()
                
                if href and text and "/resources/car-insurance/" in href and not any(x in href for x in ["quote", "bundle", "calculator"]):
                    print(f"Potential article link found:")
                    print(f"Text: {text}")
                    print(f"Href: {href}")
                    article_urls.add(href)
                    print("Added to article list\n")
                    
            except Exception as e:
                continue
                
    except Exception as e:
        print(f"Error fetching article URLs: {str(e)}")
    finally:
        driver.quit()
        
    return article_urls

def scrape_article(url: str) -> Dict:
    """Scrape a single article page."""
    print(f"\n=== Processing article: {url} ===")
    
    try:
        print("Fetching article...")
        response = requests.get(url)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else ""
            print(f"Found title: {title}")
            
            # Extract main content - adjust selector based on actual page structure
            content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            if content:
                text = content.get_text(separator=' ', strip=True)
                print(f"Found content length: {len(text)} characters")
                return {
                    "url": url,
                    "title": title,
                    "content": text[:1000]  # Limit content for testing
                }
            else:
                print("No content found in article")
    except Exception as e:
        print(f"Error scraping article: {str(e)}")
    
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