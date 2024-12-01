from pathlib import Path
import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import warnings

warnings.filterwarnings("ignore")

# Load secrets from a JSON configuration file
def read_secrets(config_path):
    with open(config_path) as config_file:
        secrets = json.load(config_file)
        return {"OPENAI_API_KEY": secrets["OPENAI_API_KEY"]}

# Apply API keys as environment variables
def configure_environment(secrets):
    os.environ['OPENAI_API_KEY'] = secrets["OPENAI_API_KEY"]

# Function to load and split documents
def document_loader(file_path):
    doc_loader = UnstructuredMarkdownLoader(file_path)
    loaded_pages = doc_loader.load()
    print(f"Successfully loaded {len(loaded_pages)} pages from {file_path}")

    # Split document content into manageable chunks
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=300,
        chunk_overlap=50
    )
    split_texts = splitter.split_documents(loaded_pages)
    return split_texts

# Function to initialize or create a Chroma vector store
def setup_vectorstore(db_name):
    storage_path = f"src/data/chroma_db/{db_name}"
    doc_file = "src/data/papers/ThrunPaper.md"

    if os.path.exists(storage_path):
        print(f"Loading existing vector store from {storage_path}")
        vector_store = Chroma(
            collection_name=db_name,
            embedding_function=OpenAIEmbeddings(),
            persist_directory=storage_path
        )
    else:
        print(f"Creating new vector store at {storage_path}")
        document_splits = document_loader(doc_file)
        vector_store = Chroma.from_documents(
            collection_name=db_name,
            documents=document_splits,
            embedding=OpenAIEmbeddings(),
            persist_directory=storage_path
        )
    return vector_store

# Function to get an answer using GPT as a fallback
def fallback_to_gpt(question):
    query_prompt = f"Answer the following question using general knowledge: {question}"
    language_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    answer = language_model(query_prompt).content
    return answer

# Function for document-based retrieval and generating responses
def answer_query_with_rag(question, db_name="papers"):
    # Check if the question is empty and return an appropriate response
    if not question.strip():  # If question is empty or just spaces
        return {
            "answer": "I cannot find information in the database.",
            "source": "general"
        }
    
    vector_store = setup_vectorstore(db_name)
    retriever = vector_store.as_retriever()

    # Retrieve relevant information from the documents
    relevant_docs = retriever.get_relevant_documents(question)

    if relevant_docs:
        # If relevant documents are found, construct a context from them
        context = "\n\n".join([doc.page_content.strip() for doc in relevant_docs if doc.page_content.strip()])
        
        # Construct a prompt using the context
        query_prompt = f"""You are a knowledgeable assistant. Use the context below to answer the question. If you don't know the answer, just say that you cannot find information in the database.
        Question: {question}
        Context: {context}
        Answer:"""
        language_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        answer = language_model(query_prompt).content
        
        # Check if the response indicates that no information was found
        if "I cannot find information in the database" in answer:
            print("Cannot find the information in Database. Generating response using GPT's general knowledge.")
            answer = fallback_to_gpt(question)  # Fallback to GPT
            response_source = "general"
        else:
            response_source = "document"
    else:
        # Use a fallback prompt with general knowledge if no documents found
        print("No relevant context found. Generating answer using general knowledge...")
        answer = fallback_to_gpt(question)  # Fallback to GPT
        response_source = "general"

    return {
        "answer": answer,
        "source": response_source,
    }

# Main execution function with detailed response source tracking
def main():
    """Main execution function with detailed response source tracking."""
    print("Welcome to the RAG-based Q&A Application!")
    try:
        config_path = Path('/home/coderpad/app/src/secrets.json')
        secrets = read_secrets(config_path)
        configure_environment(secrets)

        database_name = "papers"

        sample_queries = [
            "What is the primary goal of reinforcement learning methods?",
            "What are the current methods used in reinforcement learning to estimate the value of actions?",
            "What IPO in Investment Banking?",  # Example question not related to the document
            "What is the name of the lead actor in Avatar?"
        ]
        
        for query in sample_queries:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            result = answer_query_with_rag(query, db_name=database_name)
            
            print(f"\nAnswer: {result['answer']}")
            print(f"\nSource Information:")
            if result["source"] == "document":
                print("- Source: Retrieved from document-based context")
            else:
                print("- Source: Generated using GPT's general knowledge")
            
            print(f"\n{'='*80}")

    except Exception as e:
        print(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()