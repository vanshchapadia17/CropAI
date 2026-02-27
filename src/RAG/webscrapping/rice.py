import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin

BASE_URL = "import requests"
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin

BASE_URL = "http://www.agritech.tnau.ac.in/expert_system/sugar/"
START_PAGE = BASE_URL + "index.html"

visited = set()
pages_data = []

output_folder = "rice_dataset"
os.makedirs(output_folder, exist_ok=True)


def scrape_page(url):

    if url in visited:
        return

    visited.add(url)

    try:
        print("Scraping:", url)

        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # remove scripts
        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text(separator="\n", strip=True)

        pages_data.append({
            "url": url,
            "text": text
        })

        # save each page as txt
        file_name = url.split("/")[-1].replace(".html", "") + ".txt"
        file_path = os.path.join(output_folder, file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Source: {url}\n\n")
            f.write(text)

        # find internal links
        for link in soup.find_all("a", href=True):

            href = link["href"]

            if ".html" in href:
                full_url = urljoin(BASE_URL, href)

                if BASE_URL in full_url:
                    scrape_page(full_url)

    except Exception as e:
        print("Error:", e)


scrape_page(START_PAGE)

print("\nTotal pages scraped:", len(pages_data))
START_PAGE = BASE_URL + "index.html"

visited = set()
pages_data = []

output_folder = "rice_dataset"
os.makedirs(output_folder, exist_ok=True)


def scrape_page(url):

    if url in visited:
        return

    visited.add(url)

    try:
        print("Scraping:", url)

        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # remove scripts
        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text(separator="\n", strip=True)

        pages_data.append({
            "url": url,
            "text": text
        })

        # save each page as txt
        file_name = url.split("/")[-1].replace(".html", "") + ".txt"
        file_path = os.path.join(output_folder, file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Source: {url}\n\n")
            f.write(text)

        # find internal links
        for link in soup.find_all("a", href=True):

            href = link["href"]

            if ".html" in href:
                full_url = urljoin(BASE_URL, href)

                if BASE_URL in full_url:
                    scrape_page(full_url)

    except Exception as e:
        print("Error:", e)


scrape_page(START_PAGE)

print("\nTotal pages scraped:", len(pages_data))