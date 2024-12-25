from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from dotenv import load_dotenv
import pinecone
import os

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("allstate-articles")

# Initialize OpenAI
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4-1106-preview",
    temperature=0.7
)

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Create Pinecone as a vector store
from langchain.vectorstores import Pinecone
vectorstore = Pinecone(index, embeddings.embed_query, "text")

# Set up the prompt template for combining documents
COMBINE_PROMPT_TEMPLATE = """Given the following extracted parts of articles about insurance and a question, create a comprehensive answer with citations.

Question: {question}

Relevant article sections:
{summaries}

Instructions:
1. Use only the information from the provided article sections
2. Cite sources using [Title](URL) format after relevant information
3. If you don't know something, say so
4. End with a "Sources Used" section listing all unique sources
5. Be concise but thorough

Answer:"""

combine_prompt = PromptTemplate(
    template=COMBINE_PROMPT_TEMPLATE,
    input_variables=["summaries", "question"]
)

def get_answer(question: str) -> str:
    """Get an answer from the QA system."""
    try:
        # Get relevant documents first
        docs = vectorstore.similarity_search(question, k=3)
        print("\nRetrieved Documents:")
        for i, doc in enumerate(docs, 1):
            print(f"\nDocument {i}:")
            print(f"Source: {doc.metadata.get('url', 'No URL')}")
            print(f"Title: {doc.metadata.get('title', 'No Title')}")
            print(f"Content Preview: {doc.page_content[:200] if doc.page_content else 'No content'}")
        
        # Format documents for better source tracking
        formatted_docs = []
        for doc in docs:
            if doc.page_content:
                # Create document with required source metadata
                formatted_docs.append(Document(
                    page_content=doc.metadata.get('text', doc.page_content),
                    metadata={
                        'source': doc.metadata.get('url', 'No URL'),
                        'title': doc.metadata.get('title', 'No Title')
                    }
                ))
        
        if not formatted_docs:
            return "I couldn't find any relevant information in my knowledge base to answer your question."
        
        # Create a QA chain with the correct prompt
        qa_chain = load_qa_with_sources_chain(
            llm=llm,
            chain_type="stuff",
            prompt=combine_prompt
        )
        
        # Combine all document content
        summaries = "\n\n".join(doc.page_content for doc in formatted_docs)
        
        # Get the answer using the formatted documents
        result = qa_chain(
            {"input_documents": formatted_docs, "question": question, "summaries": summaries},
            return_only_outputs=True
        )
        
        return result["output_text"]
    except Exception as e:
        return f"Error getting answer: {str(e)}"

def main():
    print("Insurance QA Bot (type 'quit' to exit)")
    print("--------------------------------------")
    
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ['quit', 'exit']:
            break
            
        if question:
            answer = get_answer(question)
            print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
