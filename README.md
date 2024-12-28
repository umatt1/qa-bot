# Insurance QA Chatbot

An intelligent chatbot powered by LangChain that can answer insurance-related questions using Allstate's knowledge base.

## Project Structure

### Data Ingestion (`ingest.py`)
- Web scraping script that extracts articles from Allstate's resources
- Uses Selenium for handling JavaScript-rendered content
- Processes and stores article content in Pinecone vector database
- Generates embeddings using OpenAI's embedding model

#### Article Discovery Process
The ingestion script finds articles through the following process:
1. Locates the main content area of resource pages
2. Searches for article sections using specific CSS selectors
3. Falls back to list items if no article sections are found
4. Filters articles based on URL patterns and content relevance

Current filtering criteria:
- Must contain "/resources/car-insurance/" in URL
- Excludes URLs containing: quotes, bundles, calculators, etc.
- Excludes generic navigation links
- Validates both link text and URL are present

#### Known Limitations
1. Article Discovery:
   - Currently limited to 3 articles per page (configurable)
   - Some valid articles may be missed due to strict filtering
   - Dependent on specific HTML structure patterns

2. Content Processing:
   - Handles JavaScript-rendered content with wait times
   - Chunks content into 500-character segments
   - Uses recursive text splitting with 50-character overlap

#### Planned Improvements
1. Enhanced Article Discovery:
   - Increase maximum article limit
   - Refine HTML structure analysis
   - Optimize link filtering criteria

2. Content Processing:
   - Improved chunk size optimization
   - Better handling of article metadata
   - Enhanced error recovery

### QA Bot (`qa_bot.py`)
- Implements an intelligent QA system using LangChain and GPT-4
- Performs context-aware retrieval using RAG (Retrieval Augmented Generation)
- Provides source citations for answers
- Supports natural language queries about insurance topics
- Uses Pinecone for efficient similarity search of relevant content

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
```

3. Run the ingestion script:
```bash
python ingest.py
```

## Technical Details

### Vector Database
- Uses Pinecone for storing article embeddings
- Each document is chunked and embedded using OpenAI's embeddings
- Metadata includes article URL, title, and content preview

### QA System Architecture
- RAG (Retrieval Augmented Generation) for accurate responses
- LangChain for orchestrating the QA pipeline
- Pinecone vector similarity search for relevant context retrieval
- GPT-4 for natural language understanding and response generation
- Source citation system for transparency

## Future Development
1. Add support for multi-turn conversations with memory
2. Expand coverage to more insurance topics and knowledge bases
3. Implement additional fact-checking mechanisms
4. Add support for policy-specific questions
5. Enhance source citation formatting and presentation

## Contributing
Feel free to contribute by opening issues or submitting pull requests.

## License
[MIT License](LICENSE)
