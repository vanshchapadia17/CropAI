import os
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests
import urllib3
from bs4 import BeautifulSoup

# TNAU's SSL cert is expired. Public-text scrape — no MITM concern worth a crash.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# TNAU sugarcane sources. Each START_PAGE seeds a crawl; we recurse into any
# .html on the same host that carries a sugar/cane marker in its URL, so the
# scrape stays scoped to sugarcane even across directories like /agriculture/
# and /crop_protection/.
START_PAGES = [
    "http://www.agritech.tnau.ac.in/agriculture/sugarcrops_sugarcane.html",
    "https://agritech.tnau.ac.in/crop_protection/"
    "crop_prot_crop%20diseases_cash%20crops_sugarcane.html",
]
ALLOWED_HOST_SUFFIX = "agritech.tnau.ac.in"
SUGAR_MARKERS = ("sugar", "cane")

visited = set()
pages_data = []

# Always write to src/RAG/sugarcane_dataset/ regardless of where this is invoked.
output_folder = str(Path(__file__).resolve().parent.parent / "sugarcane_dataset")
os.makedirs(output_folder, exist_ok=True)


def normalize(url: str) -> str:
    """Force https + strip www. so http/https duplicates resolve to one key."""
    parsed = urlparse(url)
    netloc = parsed.netloc
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path or "/"
    return f"https://{netloc}{path}"


def should_follow(url: str) -> bool:
    """Same-host .html pages that carry a sugar/cane marker in the URL."""
    parsed = urlparse(url)
    if not parsed.netloc.endswith(ALLOWED_HOST_SUFFIX):
        return False
    if not url.lower().endswith(".html"):
        return False
    lowered = url.lower()
    return any(marker in lowered for marker in SUGAR_MARKERS)


def url_to_filename(url: str) -> str:
    """Turn a URL into a human-readable .txt filename."""
    last = unquote(url.split("/")[-1])
    if last.lower().endswith(".html"):
        last = last[:-5]
    last = "_".join(last.split())
    return last + ".txt"


def scrape_page(url: str) -> None:
    key = normalize(url)
    if key in visited:
        return
    visited.add(key)

    try:
        print("Scraping:", url)
        response = requests.get(url, timeout=15, verify=False)
        response.raise_for_status()
        # TNAU sends utf-8 bytes with a latin-1 content-type; fix before decoding.
        response.encoding = response.apparent_encoding or "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style"]):
            tag.extract()

        text = soup.get_text(separator="\n", strip=True)
        pages_data.append({"url": url, "text": text})

        file_path = os.path.join(output_folder, url_to_filename(url))
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Source: {url}\n\n")
            f.write(text)

        for link in soup.find_all("a", href=True):
            full_url = urljoin(url, link["href"]).split("#", 1)[0]
            if should_follow(full_url):
                scrape_page(full_url)

    except Exception as e:
        print(f"Error scraping {url}: {e}")


if __name__ == "__main__":
    for start in START_PAGES:
        scrape_page(start)
    print(f"\nTotal pages scraped: {len(pages_data)}")
    print(f"Saved to: {output_folder}")
