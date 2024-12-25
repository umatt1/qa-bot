# Insurance QA Chatbot

An intelligent chatbot powered by LangChain that can answer insurance-related questions using Allstate's knowledge base.

## Project Structure

### Data Ingestion (`ingest.py`)
- Web scraping script that extracts articles from Allstate's resources
- Uses Selenium for handling JavaScript-rendered content
- Processes and stores article content in Pinecone vector database
- Generates embeddings using OpenAI's embedding model

### Planned Features
- LangChain/LangGraph-based QA system
- Natural language query processing
- Context-aware responses using RAG (Retrieval Augmented Generation)
- Support for complex insurance-related queries
- Memory for maintaining conversation context

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

### Planned Architecture
- RAG (Retrieval Augmented Generation) for accurate responses
- LangChain for orchestrating the conversation flow
- LangGraph for complex multi-step reasoning
- Vector similarity search for relevant context retrieval

## Future Development
1. Implement the QA system using LangChain/LangGraph
2. Add support for multi-turn conversations
3. Implement fact-checking and source citation
4. Add support for policy-specific questions
5. Integrate with more insurance knowledge bases

## Contributing
Feel free to contribute by opening issues or submitting pull requests.

## License
[MIT License](LICENSE)
