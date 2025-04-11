# test_parse.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# test_parse.py
import pytest
from unittest.mock import patch, MagicMock
from parse import ParseWithOllama
from langchain_core.documents import Document

def test_format_docs():
    # Create an instance of ParseWithOllama.
    parser = ParseWithOllama()
    # Prepare a list of Document objects.
    docs = [
        Document(page_content="First piece of content"),
        Document(page_content="Second piece of content")
    ]
    result = parser.format_docs(docs)
    # Assert that the contents are joined with "\n\n".
    assert result == "First piece of content\n\nSecond piece of content"

@patch('parse.ChatPromptTemplate')
@patch('parse.Chroma')
def test_parse_with_ollama(mock_chroma, mock_prompt):
    # Create an instance of ParseWithOllama.
    parser = ParseWithOllama()
    
    # Set up a dummy response that our chain should return.
    dummy_response = "parsed output"
    
    # Create a dummy chain (simulated via a MagicMock) with an invoke method.
    dummy_chain = MagicMock()
    dummy_chain.invoke.return_value = dummy_response
    
    # Here is the key change: make dummy_chain chainable by having its __or__ return itself.
    dummy_chain.__or__.return_value = dummy_chain

    # Simulate ChatPromptTemplate.from_template returning an object that participates
    # in chain composition using the "|" operator.
    dummy_prompt = MagicMock()
    # When the dummy prompt is composed with the model, return the dummy_chain.
    dummy_prompt.__or__.return_value = dummy_chain
    mock_prompt.from_template.return_value = dummy_prompt

    # Setup a dummy vector store that will be returned by Chroma.from_texts.
    dummy_doc = Document(page_content="dummy content")
    dummy_vectorstore = MagicMock()
    dummy_vectorstore.similarity_search.return_value = [dummy_doc]
    mock_chroma.from_texts.return_value = dummy_vectorstore

    # Define inputs for parse_with_ollama.
    dom_chunks = ["chunk1", "chunk2"]
    parse_description = "dummy description"

    # Call the method under test.
    result = parser.parse_with_ollama(dom_chunks, parse_description)
    
    # Verify that the formatted content is built correctly from the similarity search.
    expected_formatted_content = parser.format_docs([dummy_doc])
    dummy_chain.invoke.assert_called_with({
        "dom_content": expected_formatted_content,
        "parse_description": parse_description
    })
    # Finally, check that the response from parse_with_ollama matches our dummy response.
    assert result == dummy_response
