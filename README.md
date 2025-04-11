Inspired by https://github.com/techwithtim/AI-Web-Scraper/tree/main

`Note`: **If you don't have access to BrighData services, you'll not be able to use this app.**

Added RAG to speed up the parsing process. Dockerized and ci/cd github actions added.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
brightdata webdriver used for better handling of captcha

3. Run the app:
```bash
streamlit run main.py
```
