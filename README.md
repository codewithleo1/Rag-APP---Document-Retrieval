# RAG-based Q&A System

This project implements a Retrieval-Augmented Generation (RAG) system that combines document retrieval and generative AI to answer user queries. It uses FAISS for storing and retrieving document embeddings, OpenAI's GPT-3.5 model for generating answers, and encryption mechanisms to securely handle API keys.

## Features

- **Document Retrieval:** The system retrieves relevant content from a set of documents (stored in FAISS index) to answer user questions.
- **Fallback to General Knowledge:** If no relevant documents are found, the system falls back to generating an answer using general knowledge from GPT-3.5.
- **Encryption:** API keys and secrets are securely stored in encrypted files and decrypted at runtime.
- **Streamlit Interface:** A simple chat interface that allows users to ask questions and receive answers.
- **Sample Questions:** Pre-defined sample questions to test the system.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/rag-qa-system.git
   cd rag-qa-system
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Obtain your OpenAI API key and save it in a secure location.

4. Store the API key in the `secrets_encrypted.json` file. The encryption process is handled using the `cryptography` library.

5. Ensure that the document(s) for indexing are available and specify their paths in the script.

## File Structure

- **faiss_indices/**: Contains the FAISS index and metadata files.
- **src/**: Source code directory.
  - **data/**: Directory for app data, documents, and database files.
  - **app.py**: Main entry point for running the Streamlit app.
  - **encrypt.py**: Script for encrypting the API secrets file.
  - **main.py**: Core logic for document processing and question answering.
  - **secret.key**: Encryption key used for decrypting the API secrets.
  - **secrets.json**: Plain-text API secrets (to be encrypted before use).
  - **secrets_encrypted.json**: Encrypted secrets file.
  - **tests.py**: Unit tests for the system components.
  - **requirements.txt**: List of required Python packages.

## Usage

1. **Running the Application:**

   To start the Streamlit application, run:

   ```bash
   streamlit run src/app.py
   ```

   This will launch the Q&A system in your browser.

2. **Asking Questions:**
   - Type your question in the input field and click "Ask Question".
   - The system will either retrieve relevant information from the documents or fall back to general knowledge if no relevant content is found.

3. **Sample Questions:**
   - You can try predefined sample questions for testing.
   - These questions are listed on the right column of the interface.

## How It Works

1. **Secrets Management:**
   - The API key used for OpenAI's GPT model is securely managed using encryption. The secret key (`secret.key`) is used to decrypt the `secrets_encrypted.json` file at runtime.

2. **Document Loading and Indexing:**
   - Documents are loaded from markdown files and split into smaller chunks using Langchain's `RecursiveCharacterTextSplitter`.
   - These chunks are then converted into embeddings using OpenAI's `OpenAIEmbeddings`, and stored in a FAISS index for efficient retrieval.

3. **RAG (Retrieval-Augmented Generation):**
   - When a question is asked, the system retrieves relevant document chunks from the FAISS index.
   - If relevant documents are found, the system uses the context from those documents to generate a response using OpenAI's `ChatOpenAI` model.
   - If no relevant documents are found, the system generates a general answer using GPT-3.5.

4. **Streamlit Interface:**
   - The user interface is built using Streamlit, allowing users to interact with the system through a simple chat interface.

## Security Considerations

- **API Key Security:** The OpenAI API key is stored in an encrypted file (`secrets_encrypted.json`) and is decrypted at runtime using the encryption key stored in `secret.key`.
- **Document Privacy:** The documents loaded for the FAISS index are not shared outside the local environment. Make sure that the documents used are safe to handle within your environment.

## Dependencies

The project requires the following Python libraries:

- **Streamlit**: For the web interface.
- **Langchain**: For document processing, embedding, and retrieval.
- **OpenAI**: For integrating GPT-3.5 model for query answering.
- **FAISS**: For fast and efficient vector storage and retrieval.
- **Cryptography**: For encrypting and decrypting sensitive files.
- **Pandas, NumPy, and other utilities**: For data handling.

Install them by running:

```bash
pip install -r requirements.txt
```

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Langchain**: For building the infrastructure for document splitting, embeddings, and retrieval.
- **OpenAI**: For the GPT-3.5 model, which powers the generative aspect of the system.
- **FAISS**: For fast and efficient vector storage and retrieval.
