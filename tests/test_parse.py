# test_parse.py
import os
import sys

# Insert the parent directory (project root) into sys.path so that modules in web_scrape can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from web_scrape.parse import ParseWithOllama


def test_format_docs():
    parser = ParseWithOllama()
    docs = [
        Document(page_content="First piece of content"),
        Document(page_content="Second piece of content")
    ]
    result = parser.format_docs(docs)
    assert result == "First piece of content\n\nSecond piece of content"

@patch('web_scrape.parse.ChatPromptTemplate')
@patch('web_scrape.parse.Chroma')
def test_parse_with_ollama(mock_chroma, mock_prompt):
    parser = ParseWithOllama()
    dummy_response = "parsed output"

    dummy_chain = MagicMock()
    dummy_chain.invoke.return_value = dummy_response
    dummy_chain.__or__.return_value = dummy_chain

    dummy_prompt = MagicMock()
    dummy_prompt.__or__.return_value = dummy_chain
    mock_prompt.from_template.return_value = dummy_prompt

    dummy_doc = Document(page_content="dummy content")
    dummy_vectorstore = MagicMock()
    dummy_vectorstore.similarity_search.return_value = [dummy_doc]
    mock_chroma.from_texts.return_value = dummy_vectorstore

    dom_chunks = ["chunk1", "chunk2"]
    parse_description = "dummy description"

    result = parser.parse_with_ollama(dom_chunks, parse_description)
    
    expected_formatted_content = parser.format_docs([dummy_doc])
    dummy_chain.invoke.assert_called_with({
        "dom_content": expected_formatted_content,
        "parse_description": parse_description
    })
    assert result == dummy_response
