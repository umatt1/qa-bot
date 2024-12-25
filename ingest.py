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
from webdriver_manager.firefox import GeckoDriverManager

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
    
    try:
        service = Service(GeckoDriverManager().install())
        driver = webdriver.Firefox(service=service, options=firefox_options)
        return driver
    except Exception as e:
        print(f"Error setting up Firefox driver: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

def get_article_urls(base_urls):
    article_urls = set()
    driver = setup_driver()
    
    try:
        for base_url in base_urls:
            try:
                print(f"\n=== Fetching from {base_url} ===")
                
                # Load the page with Selenium
                driver.get(base_url)
                print("Page loaded, waiting for content...")
                
                # Wait for the main content to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "main"))
                )
                
                # Give extra time for dynamic content to load
                time.sleep(2)
                
                # Get the page source after JavaScript has run
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                
                print(f"\nPage Title: {soup.title.string if soup.title else 'No title found'}")
                
                # Find all article links
                print("\nLooking for article links...")
                links = soup.find_all('a', href=True)
                print(f"Found {len(links)} total links")
                
                for link in links:
                    href = link['href']
                    text = link.text.strip()
                    
                    # Only process links that look like articles
                    if '/resources/' in href and href != base_url and text:
                        print(f"\nPotential article link found:")
                        print(f"Text: {text}")
                        print(f"Href: {href}")
                        
                        # Make sure we have absolute URLs
                        if not href.startswith('http'):
                            href = urljoin(base_url, href)
                        
                        article_urls.add(href)
                        print("Added to article list")
                
                print(f"\nFound {len(article_urls)} unique articles so far")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error processing {base_url}: {str(e)}")
                continue
    
    finally:
        driver.quit()
    
    print(f"\nTotal unique article URLs collected: {len(article_urls)}")
    return list(article_urls)

def scrape_article(url):
    """Scrape an individual article."""
    driver = setup_driver()
    
    try:
        print(f"\n=== Processing article: {url} ===")
        
        # Load the page with Selenium
        driver.get(url)
        print("Page loaded, waiting for content...")
        
        # Wait for content to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "main"))
        )
        
        # Give extra time for dynamic content to load
        time.sleep(2)
        
        # Get the page source after JavaScript has run
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        print("\nArticle page structure:")
        print("Title tag content:", soup.title.string if soup.title else "No title found")
        
        # Try different content selectors
        content_selectors = [
            ('div', 'article-content'),
            ('div', 'article-body'),
            ('main', None),
            ('article', None),
            ('div', lambda x: x and ('content' in x.lower() or 'article' in x.lower()))
        ]
        
        content = None
        for tag, class_ in content_selectors:
            content = soup.find(tag, class_=class_)
            if content:
                print(f"\nFound content using selector: {tag}, {class_}")
                break
        
        if not content:
            print("No article content found with any selector")
            return None
        
        # Extract title
        title = None
        title_elements = soup.find_all(['h1', 'h2'], class_=lambda x: x and ('title' in x.lower() or 'heading' in x.lower()))
        if title_elements:
            title = title_elements[0].get_text().strip()
        if not title and soup.find('h1'):
            title = soup.find('h1').get_text().strip()
        
        # Extract content
        paragraphs = content.find_all(['p', 'h2', 'h3', 'h4', 'li'])
        article_text = '\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        if not article_text:
            print("No article text found")
            return None
        
        print(f"\nExtracted content preview: {article_text[:200]}...")
        
        return {
            'url': url,
            'title': title or "Untitled Article",
            'content': article_text
        }
    
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None
    
    finally:
        driver.quit()

def process_and_upload_articles(article_urls):
    """Process articles and upload them to Pinecone."""
    # Create index if it doesn't exist
    if index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embedding dimension
            metric='cosine'
        )
    
    index = pc.Index(index_name)
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    
    for url in article_urls:
        article = scrape_article(url)
        if not article:
            continue
        
        print(f"\nProcessing article: {article['title']}")
        
        # Split text into chunks
        chunks = text_splitter.split_text(article['content'])
        
        # Create metadata for each chunk
        metadatas = [{
            'url': article['url'],
            'title': article['title'],
            'chunk': i,
            'text': chunk[:200]  # Store preview of the chunk
        } for i, chunk in enumerate(chunks)]
        
        # Get embeddings for chunks
        try:
            chunk_embeddings = embeddings.embed_documents(chunks)
            
            # Prepare vectors for upload
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                vector_id = f"{article['url']}_{i}"
                vectors.append((vector_id, embedding, metadatas[i]))
            
            # Upload vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                index.upsert(vectors=batch)
                print(f"Uploaded batch {i//batch_size + 1} of {(len(vectors)-1)//batch_size + 1}")
            
        except Exception as e:
            print(f"Error processing chunks for {article['title']}: {str(e)}")
            continue
        
        print(f"Completed processing article: {article['title']}")

if __name__ == "__main__":
    base_urls = [
        "https://www.allstate.com/resources/car-insurance",
        "https://www.allstate.com/resources/home-insurance",
        "https://www.allstate.com/resources/financial",
        "https://www.allstate.com/resources/life-insurance",
        "https://www.allstate.com/resources/retirement",
    ]
    article_urls = get_article_urls(base_urls)
    process_and_upload_articles(article_urls)