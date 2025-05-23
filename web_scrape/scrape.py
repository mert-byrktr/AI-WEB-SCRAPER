import os
import time

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from selenium.common.exceptions import WebDriverException
from selenium.webdriver import ChromeOptions, Remote
from selenium.webdriver.chromium.remote_connection import \
    ChromiumRemoteConnection

load_dotenv("sample.env")

SBR_WEBDRIVER = os.getenv("SBR_WEBDRIVER")

def scrape_website(website):

    print("Connecting to Scraping Browser...")
    sbr_connection = ChromiumRemoteConnection(SBR_WEBDRIVER, "goog", "chrome")
    with Remote(sbr_connection, options=ChromeOptions()) as driver:
        print(f"Connected! Navigation to the {website}")
        driver.get(website)
        print("Waiting captcha to solve...")
        solve_res = driver.execute(
            "executeCdpCommand",
            {
                "cmd": "Captcha.waitForSolve",
                "params": {"detectTimeout": 10000},
            },
        )
        print("Captcha solve status:", solve_res["value"]["status"])
        print("Navigated! Scraping page content...")
        retries = 3
        for i in range(retries):
            try:
                html = driver.page_source
                break
            except WebDriverException as e:
                print(f"Attempt {i+1} failed: {e}")
                time.sleep(3)
                if i == retries -1:
                    raise
        return html


def extract_body_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    body_content = soup.body
    if body_content:
        return str(body_content)
    return ""


def clean_body_content(body_content):
    soup = BeautifulSoup(body_content, "html.parser")

    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()

    # Get text or further process the content
    cleaned_content = soup.get_text(separator="\n")
    cleaned_content = "\n".join(
        line.strip() for line in cleaned_content.splitlines() if line.strip()
    )

    return cleaned_content


def split_dom_content(dom_content, max_length=6000):
    return [
        dom_content[i : i + max_length] for i in range(0, len(dom_content), max_length)
    ]