from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
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

# Set up the prompt template
qa_template = """You are an expert insurance advisor chatbot trained on Allstate's knowledge base. Your goal is to provide accurate, helpful information about insurance topics.

Context: {context}

Current conversation:
{chat_history}

Human Question: {question}

Please provide a helpful, accurate response based on the context provided. If you're unsure about something, say so rather than making assumptions. Always cite specific information from the context when possible."""

QA_PROMPT = PromptTemplate(
    template=qa_template,
    input_variables=["context", "chat_history", "question"]
)

# Set up conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create the conversation chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT}
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
        
        # Get the answer
        result = qa({"question": question})
        return result['answer']
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
