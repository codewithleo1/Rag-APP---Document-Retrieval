import os
import json
import unittest
from main import answer_query_with_rag

# Load secrets from a JSON configuration file
def read_secrets(config_path):
    with open(config_path) as config_file:
        secrets = json.load(config_file)
        return {"OPENAI_API_KEY": secrets["OPENAI_API_KEY"]}

# Apply API keys as environment variables
def configure_environment(secrets):
    os.environ['OPENAI_API_KEY'] = secrets["OPENAI_API_KEY"]

class TestRAGApplication(unittest.TestCase):
    
    def setUp(self):
        self.db_name = "papers"  # Replace with your actual database name if needed
        
        # Load secrets and configure the environment
        config_path = 'src/secrets.json'  # Update with your actual path to secrets.json
        secrets = read_secrets(config_path)
        configure_environment(secrets)

    def tearDown(self):
        print("="*80)  # Print a separator line after each test

    def test_valid_query_document(self):
        # Test a valid query that is expected to return document-based information
        question = "What is the primary goal of reinforcement learning methods?"
        print(f"Testing question: {question}")  # Print the question
        result = answer_query_with_rag(question, db_name=self.db_name)
        
        self.assertIsInstance(result, dict)
        self.assertIn('answer', result)
        self.assertIn('source', result)
        self.assertEqual(result['source'], "document")
        self.assertIsNotNone(result['answer'])  # Ensure the answer is not None

    def test_valid_query_general(self):
        # Test a valid query that is expected to return a general knowledge answer
        question = "What IPO in Investment Banking?"
        print(f"Testing question: {question}")  # Print the question
        result = answer_query_with_rag(question, db_name=self.db_name)
        
        self.assertIsInstance(result, dict)
        self.assertIn('answer', result)
        self.assertIn('source', result)
        self.assertEqual(result['source'], "general")
        self.assertIsNotNone(result['answer'])  # Ensure the answer is not None

    def test_empty_query(self):
        # Test handling of an empty query
        question = ""
        print(f"Testing question: {question}")  # Print the question
        result = answer_query_with_rag(question, db_name=self.db_name)

        print(f"Result for empty query: {result}")  # Debugging output

        self.assertIsInstance(result, dict)
        self.assertIn('answer', result)
        self.assertIn('source', result)
        self.assertEqual(result['answer'], "I cannot find information in the database.")  # Handle empty query correctly
        self.assertEqual(result['source'], "general")

    def test_invalid_query(self):
        # Test handling of a completely invalid query
        question = "What is the capital of the Moon?"
        print(f"Testing question: {question}")  # Print the question
        result = answer_query_with_rag(question, db_name=self.db_name)

        self.assertIsInstance(result, dict)
        self.assertIn('answer', result)
        self.assertIn('source', result)
        self.assertEqual(result['source'], "general")  # Assuming this will not match the document
        self.assertIsNotNone(result['answer'])  # Ensure the answer is not None

if __name__ == "__main__":
    unittest.main()