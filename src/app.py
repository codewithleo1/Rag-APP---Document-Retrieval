import streamlit as st
from pathlib import Path
import json
import os
from cryptography.fernet import Fernet
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import warnings

warnings.filterwarnings("ignore")

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def read_secrets(config_path):
    # Load the encryption key
    with open("secret.key", "rb") as key_file:
        key = key_file.read()
    
    cipher_suite = Fernet(key)

    # Decrypt the secrets file
    with open(config_path, "rb") as encrypted_file:
        encrypted_data = encrypted_file.read()
        decrypted_data = cipher_suite.decrypt(encrypted_data)
        
    # Load the decrypted JSON data
    secrets = json.loads(decrypted_data.decode("utf-8"))
    return {"OPENAI_API_KEY": secrets["OPENAI_API_KEY"]}

def configure_environment(secrets):
    os.environ['OPENAI_API_KEY'] = secrets["OPENAI_API_KEY"]

def document_loader(file_path):
    doc_loader = UnstructuredMarkdownLoader(file_path)
    loaded_pages = doc_loader.load()
    
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=300,
        chunk_overlap=50
    )
    split_texts = splitter.split_documents(loaded_pages)
    return split_texts

def create_faiss_index(document_splits):
    # Initialize the OpenAI Embeddings instance
    embedding_function = OpenAIEmbeddings()

    # Create the FAISS index using the document splits and the embedding function
    vector_store = FAISS.from_documents(document_splits, embedding_function)

    return vector_store

def setup_vectorstore(db_name):
    storage_path = rf"C:\Users\Dell\Desktop\Coderpad_udacity\src\data\faiss_db/{db_name}"  # Path for FAISS storage
    doc_file = r"C:\Users\Dell\Desktop\Coderpad_udacity\src\data\papers\ThrunPaper.md"  # Use raw string

    # Load existing vector store or create a new one
    embedding_function = OpenAIEmbeddings()  # Initialize embeddings to pass when loading
    if os.path.exists(storage_path):
        # Load existing FAISS index with embeddings and allow dangerous deserialization
        vector_store = FAISS.load_local(storage_path, embedding_function, allow_dangerous_deserialization=True)
    else:
        # Load documents and create FAISS index
        document_splits = document_loader(doc_file)
        vector_store = create_faiss_index(document_splits)  # Create index with embeddings
        vector_store.save_local(storage_path)  # Save the index

    st.session_state.vector_store = vector_store
    return vector_store

def fallback_to_gpt(question):
    query_prompt = f"Answer the following question using general knowledge: {question}"
    language_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    answer = language_model.invoke(query_prompt).content
    return answer

def answer_query_with_rag(question, db_name="papers"):
    if not question.strip():
        return {
            "answer": "Please enter a valid question.",
            "source": "system"
        }
    
    with st.spinner('Searching for relevant information...'):
        vector_store = setup_vectorstore(db_name)
        retriever = vector_store.as_retriever()
        relevant_docs = retriever.get_relevant_documents(question)

    if relevant_docs:
        context = "\n\n".join([doc.page_content.strip() for doc in relevant_docs if doc.page_content.strip()])
        
        query_prompt = f"""You are a knowledgeable assistant. Use the context below to answer the question. If you don't know the answer, just say that you cannot find information in the database.
        Question: {question}
        Context: {context}
        Answer:"""
        
        with st.spinner('Generating response...'):
            language_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            answer = language_model.invoke(query_prompt).content
        
        if "I cannot find information in the database" in answer:
            with st.spinner('Generating response using general knowledge...'):
                answer = fallback_to_gpt(question)
                response_source = "general"
        else:
            response_source = "document"
    else:
        with st.spinner('Generating response using general knowledge...'):
            answer = fallback_to_gpt(question)
            response_source = "general"

    return {
        "answer": answer,
        "source": response_source,
    }

def main():
    st.set_page_config(page_title="RAG Q&A System", page_icon="🤖", layout="wide")
    
    st.title("📚 RAG-based Q&A System")
    st.markdown("""
    This application uses RAG (Retrieval-Augmented Generation) to answer your questions.
    It first searches through a document database and if no relevant information is found,
    it falls back to general knowledge.
    """)

    try:
        # Configure API keys
        config_path = Path(r'C:\Users\Dell\Desktop\Coderpad_udacity\src\secrets_encrypted.json')  # Use raw string
        secrets = read_secrets(config_path)
        configure_environment(secrets)

        # Create two columns
        col1, col2 = st.columns([2, 1])

        with col1:
            # Main chat interface
            st.subheader("Chat Interface")
            user_question = st.text_input("Enter your question:", key="user_input")
            
            if st.button("Ask Question", key="ask_button"):
                if user_question:
                    result = answer_query_with_rag(user_question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": result["answer"],
                        "source": result["source"]
                    })

                    # Display the latest response immediately
                    with st.container():
                        st.markdown("---")
                        st.markdown("**Question:** " + user_question)
                        st.markdown("**Answer:** " + result["answer"])
                        source_label = "Retrieved from document" if result["source"] == "document" else "Generated using general knowledge"
                        st.markdown(f"*Source: {source_label}*")

            # Display chat history
            for chat in reversed(st.session_state.chat_history):
                with st.container():
                    st.markdown("---")
                    st.markdown("**Question:** " + chat["question"])
                    st.markdown("**Answer:** " + chat["answer"])
                    source_label = "Retrieved from document" if chat["source"] == "document" else "Generated using general knowledge"
                    st.markdown(f"*Source: {source_label}*")

        with col2:
            # Sample questions and information
            st.subheader("Sample Questions")
            sample_questions = [
                "What is the primary goal of reinforcement learning methods?",
                "What are the current methods used in reinforcement learning to estimate the value of actions?",
                "What is IPO in Investment Banking?",
                "What is the name of the lead actor in Avatar?"
            ]
            
            st.markdown("Try these sample questions:")
            for question in sample_questions:
                if st.button(question, key=f"sample_{question}"):
                    result = answer_query_with_rag(question)
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result["answer"],
                        "source": result["source"]
                    })
                    
                    # Display the latest response immediately
                    with st.container():
                        st.markdown("---")
                        st.markdown("**Question:** " + question)
                        st.markdown("**Answer:** " + result["answer"])
                        source_label = "Retrieved from document" if result["source"] == "document" else "Generated using general knowledge"
                        st.markdown(f"*Source: {source_label}*")

            # Help section
            st.subheader("Help")
            st.markdown("""
            - Enter your question in the input field and click 'Ask Question'.
            - Use the sample questions for testing.
            - The system uses documents for retrieval and general knowledge for fallback.
            """)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
