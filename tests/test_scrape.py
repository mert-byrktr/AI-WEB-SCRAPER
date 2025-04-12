# test_scrape.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import time
from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from web_scrape.scrape import (clean_body_content, extract_body_content,
                               scrape_website, split_dom_content)


def test_extract_body_content():
    # Test with HTML that includes a <body> tag.
    html = "<html><body><h1>Header</h1><p>Paragraph</p></body></html>"
    result = extract_body_content(html)
    expected = str(BeautifulSoup(html, "html.parser").body)
    assert result == expected

def test_extract_body_content_no_body():
    # Test with HTML that does not contain a <body> tag.
    html = "<html><head>Header</head></html>"
    result = extract_body_content(html)
    assert result == ""

def test_clean_body_content():
    # Provide some HTML that includes <script> and <style> tags.
    html_content = """
    <html>
      <body>
        <script>var x = 1;</script>
        <style>body { background: #fff; }</style>
        <p>Visible Text</p>
      </body>
    </html>
    """
    result = clean_body_content(html_content)
    # The cleaned content should not include text from script or style elements.
    assert "var x = 1;" not in result
    assert "background: #fff;" not in result
    assert "Visible Text" in result

def test_split_dom_content():
    # Create a content string longer than the specified max_length.
    content = "a" * 10000
    # Split into chunks with max_length set to 6000.
    chunks = split_dom_content(content, max_length=6000)
    # Expect 2 chunks: one with 6000 characters and one with 4000.
    assert len(chunks) == 2
    assert chunks[0] == "a" * 6000
    assert chunks[1] == "a" * 4000

@patch('web_scrape.scrape.Remote')
@patch('web_scrape.scrape.ChromiumRemoteConnection')
def test_scrape_website(mock_connection, mock_remote):
    # Prepare a dummy driver to simulate Selenium's Remote driver.
    dummy_driver = MagicMock()
    # Set the page_source to a dummy HTML content.
    dummy_driver.page_source = "<html><body>Dummy Content</body></html>"
    # Simulate the execution of a command to solve the captcha.
    dummy_driver.execute.return_value = {"value": {"status": "solved"}}
    
    # Configure the mock for the Remote instance context manager to return our dummy driver.
    mock_remote.return_value.__enter__.return_value = dummy_driver
    
    website = "http://example.com"
    html = scrape_website(website)
    
    dummy_driver.get.assert_called_with(website)
    
    assert "Dummy Content" in html
