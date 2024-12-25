import requests
from bs4 import BeautifulSoup
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import os
from typing import List, Dict
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI and Pinecone
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-west1-gcp-free"  # Free tier environment

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "allstate-articles"

# Create serverless index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embeddings are 1536 dimensions
        metric='cosine',
        spec=pinecone.ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )
index = pinecone.Index(index_name)

def scrape_article(url: str) -> Dict[str, str]:
    """Scrape a single article from Allstate resources."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract article content (adjust selectors based on actual HTML structure)
        title = soup.find('h1').get_text().strip() if soup.find('h1') else ""
        content = ""
        article_body = soup.find('article') or soup.find('div', class_='article-content')
        if article_body:
            paragraphs = article_body.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs])
        
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
    base_url = "https://www.allstate.com/resources"
    urls = []
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all article links (adjust selector based on actual HTML structure)
        article_links = soup.find_all('a', href=True)
        for link in article_links:
            href = link['href']
            if href.startswith('/resources/'):
                full_url = f"https://www.allstate.com{href}"
                urls.append(full_url)
    except Exception as e:
        print(f"Error getting article URLs: {str(e)}")
    
    return urls

def process_and_upload_articles():
    """Main function to scrape articles and upload to Pinecone."""
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
        if not article:
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