import requests
from bs4 import BeautifulSoup
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import os
from typing import List, Dict
import time
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Initialize OpenAI and Pinecone
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "allstate-articles"

def scrape_article(url: str) -> Dict[str, str]:
    """Scrape a single article from Allstate resources."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract article content
        title = soup.find('h1').get_text().strip() if soup.find('h1') else ""
        
        # Look for content in different possible containers
        content = ""
        content_selectors = [
            'article',
            '.article-content',
            '.content-wrapper',
            'main',
            '#main-content'
        ]
        
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                # Get all paragraphs and headers
                elements = content_element.find_all(['p', 'h2', 'h3', 'h4', 'li'])
                content = ' '.join([elem.get_text().strip() for elem in elements])
                break
        
        return {
            "url": url,
            "title": title,
            "content": content
        }
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None

def get_article_urls() -> List[str]:
    """Get all article URLs from Allstate resources page."""
    base_urls = [
        "https://www.allstate.com/resources/car-insurance",
        "https://www.allstate.com/resources/home-insurance",
        "https://www.allstate.com/resources/financial",
        "https://www.allstate.com/resources/life-insurance",
        "https://www.allstate.com/resources/retirement",
    ]
    urls = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for base_url in base_urls:
        try:
            response = requests.get(base_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all links that contain /resources/
            links = soup.find_all(['a', 'link'], href=True)
            for link in links:
                href = link['href']
                if '/resources/' in href:
                    # Make sure we have absolute URLs
                    if href.startswith('/'):
                        href = f"https://www.allstate.com{href}"
                    if href not in urls:  # Avoid duplicates
                        urls.append(href)
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error getting articles from {base_url}: {str(e)}")
    
    return urls

def process_and_upload_articles():
    """Main function to scrape articles and upload to Pinecone."""
    # Create index if it doesn't exist
    if index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embeddings are 1536 dimensions
            metric='cosine'
        )
    
    index = pc.Index(index_name)
    
    # Get all article URLs
    article_urls = get_article_urls()
    print(f"Found {len(article_urls)} articles")
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    # Process each article
    for url in article_urls:
        article = scrape_article(url)
        if not article or not article['content']:
            continue
            
        print(f"Processing article: {article['title']}")
        
        # Split text into chunks
        chunks = text_splitter.split_text(article['content'])
        
        # Create embeddings and upload to Pinecone
        for i, chunk in enumerate(chunks):
            try:
                # Get embedding for the chunk
                embedding = embeddings.embed_query(chunk)
                
                # Create metadata
                metadata = {
                    "url": article['url'],
                    "title": article['title'],
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                
                # Upload to Pinecone
                index.upsert(
                    vectors=[(f"{article['url']}_{i}", embedding, metadata)],
                )
                
                # Rate limiting to avoid hitting API limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing chunk {i} of article {article['title']}: {str(e)}")
                continue
                
        print(f"Completed processing article: {article['title']}")

if __name__ == "__main__":
    process_and_upload_articles()