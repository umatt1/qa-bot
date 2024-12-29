# Insurance QA Bot

An intelligent chatbot that answers insurance-related questions using content from Auto-owners and Progressive insurance resources. The bot provides accurate, sourced information while encouraging users to contact insurance professionals for specific advice.

## Project Structure

- `ingest.py`: Scrapes and processes insurance articles from multiple sources
  - Configurable source definitions for different insurance providers
  - Automatic Pinecone index creation and management
  - Smart content extraction with anti-bot detection avoidance
  - Chunk-based document processing for better context

- `qa_bot.py`: Core question-answering logic
  - Uses OpenAI for generating human-like responses
  - Retrieves relevant context from Pinecone vector database
  - Custom prompt templates for insurance-focused answers
  - Source citation and professional guidance

- `app.py`: Streamlit-based user interface
  - Clean, modern chat interface
  - Real-time question answering
  - Chat history preservation
  - Source attribution display

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/qa-bot.git
cd qa-bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env`:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

## Usage

1. Ingest insurance articles:
```bash
python ingest.py
```
This will scrape articles from configured sources and store them in Pinecone.

2. Run the Streamlit app:
```bash
streamlit run app.py
```
The app will be available at http://localhost:8501

## Configuration

### Sources
The bot currently uses the following sources:
- Auto-owners Insurance (Primary)
- Progressive Insurance (Primary)
- Allstate (Supplementary information)

### Vector Database
- Uses Pinecone serverless (us-east-1)
- Index name: "insurance-articles"
- Embedding: OpenAI (1536 dimensions)

## Notes

- The bot is designed to provide general information and always encourages consulting with insurance professionals for specific advice
- Content is regularly updated from official insurance provider resources
- Responses include source citations for transparency
- The system avoids making specific policy recommendations

## Requirements

- Python 3.8+
- OpenAI API key
- Pinecone API key
- Firefox (for web scraping)

## License

[Your License Here]
