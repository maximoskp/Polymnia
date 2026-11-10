import json
import logging
import re
import time

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT_URL = (
    "https://hmu.gr/odigos-dioikitikon-diadikasion/"
)

OUTPUT_DIR = Path(
    "data/odigos_dioikitikon_diadikasion"
)

ARTICLES_DIR = OUTPUT_DIR / "articles"

DELAY_SECONDS = 1.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; PolymniaResearchBot/1.0; "
        "+https://hmu.gr/)"
    )
}


# --------------------------------------------------
# Logging
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=OUTPUT_DIR / "scrape.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# HTTP session
# --------------------------------------------------

session = requests.Session()
session.headers.update(HEADERS)


def fetch_page(url):
    """Download a webpage and return its BeautifulSoup object."""

    response = session.get(url, timeout=30)
    response.raise_for_status()

    response.encoding = response.apparent_encoding

    return BeautifulSoup(response.text, "lxml")

# --------------------------------------------------
# Remove navigation links
# --------------------------------------------------

def remove_guide_navigation(content):
    """
    Remove the repeated left-hand navigation menu.

    Detects the Divi column containing multiple links
    to the administrative guide articles.
    """

    root_path = urlparse(ROOT_URL).path.rstrip("/") + "/"

    def get_article_number(anchor):

        title = anchor.get_text(" ", strip=True)

        match = re.match(
            r"^Άρθρο\s+(\d+)",
            title,
            re.IGNORECASE,
        )

        if not match:
            return None

        url = urljoin(ROOT_URL, anchor["href"])
        parsed = urlparse(url)

        if parsed.netloc != urlparse(ROOT_URL).netloc:
            return None

        if not parsed.path.startswith(root_path):
            return None

        return int(match.group(1))

    # Find all navigation-related article links.
    article_links = []

    for anchor in content.find_all("a", href=True):

        number = get_article_number(anchor)

        if number is not None:
            article_links.append((anchor, number))

    # Identify Divi columns containing multiple
    # administrative article links.
    candidates = []

    for column in content.select(".et_pb_column"):

        numbers = {
            number
            for anchor, number in article_links
            if column in anchor.parents
        }

        if len(numbers) >= 5:
            candidates.append((column, len(numbers)))

    if candidates:

        # Remove the column containing the largest
        # number of article navigation links.
        column, count = max(
            candidates,
            key=lambda item: item[1],
        )

        print(
            f"    Removed navigation column "
            f"({count} article links)"
        )

        column.decompose()

    else:

        print(
            "    No navigation column detected. "
            "Keeping content unchanged."
        )

    return content

# --------------------------------------------------
# Extract navigation links
# --------------------------------------------------

def extract_article_links(soup):
    """
    Extract links pointing to individual articles
    belonging to the administrative guide.
    """

    root = urlparse(ROOT_URL)
    root_path = root.path.rstrip("/") + "/"

    articles = {}

    for anchor in soup.find_all("a", href=True):

        title = anchor.get_text(" ", strip=True)

        # Match links labelled Άρθρο 1, Άρθρο 2, etc.
        match = re.match(
            r"^Άρθρο\s+(\d+)\s*[–-]\s*(.+)$",
            title,
            re.IGNORECASE,
        )

        if not match:
            continue

        article_number = int(match.group(1))

        url = urljoin(ROOT_URL, anchor["href"])
        parsed_url = urlparse(url)

        # Only accept pages belonging to this guide.
        if parsed_url.netloc != root.netloc:
            continue

        if not parsed_url.path.startswith(root_path):
            continue

        articles[article_number] = {
            "number": article_number,
            "title": title,
            "url": url,
        }

    return sorted(
        articles.values(),
        key=lambda item: item["number"],
    )


# --------------------------------------------------
# Extract main content
# --------------------------------------------------

def extract_main_content(soup):
    """
    Extract main article content from HMU's Divi-based
    WordPress pages.
    """

    selectors = [
        "#main-content .et-l--body",
        "#main-content .et_builder_inner_content",
        ".et-l--body",
        "#main-content",
        ".et_builder_inner_content",
    ]

    content = None

    for selector in selectors:

        candidate = soup.select_one(selector)

        if candidate and len(candidate.get_text(" ", strip=True)) > 100:
            content = candidate
            print(f"    Content selector: {selector}")
            break

    if content is None:

        print("Could not find content. Inspecting Divi structure:")

        for element in soup.select(
            "#main-content, #main-content div, "
            ".et-l--body, .et_builder_inner_content"
        )[:30]:

            print(
                element.name,
                element.get("id"),
                element.get("class"),
                len(element.get_text(" ", strip=True))
            )

        raise ValueError(
            "Could not identify main content"
        )

    # Work on an independent copy.
    content = BeautifulSoup(str(content), "lxml")

    # Remove irrelevant elements.
    unwanted_selectors = [
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "iframe",
        ".et-l--header",
        ".et-l--footer",
        ".mega-menu-wrap",
        ".mega-menu",
        ".et_pb_sidebar",
        ".et_pb_post_nav",
        ".et_pb_comments",
        ".et_pb_social_media_follow",
    ]

    for selector in unwanted_selectors:
        for element in content.select(selector):
            element.decompose()

    return content


# --------------------------------------------------
# Convert content to Markdown
# --------------------------------------------------

def convert_to_markdown(content):
    """
    Convert HTML to Markdown while retaining headings,
    links, lists and tables.
    """

    md = markdownify(
        str(content),
        heading_style="ATX",
        bullets="-",
    )

    # Normalize excessive blank lines.
    md = re.sub(r"\n{3,}", "\n\n", md)

    return md.strip()


# --------------------------------------------------
# Save individual article
# --------------------------------------------------

def scrape_article(article):

    number = article["number"]
    url = article["url"]
    title = article["title"]

    filename = f"arthro_{number:02d}"

    logger.info("Scraping: %s", url)

    soup = fetch_page(url)

    content = extract_main_content(soup)

    content = remove_guide_navigation(content)

    markdown_content = convert_to_markdown(content)

    # Add metadata to the Markdown document.
    markdown_document = f"""---
title: "{title}"
article_number: {number}
source: "{url}"
scraped_at: "{datetime.now(timezone.utc).isoformat()}"
---

# {title}

{markdown_content}
"""

    # Save Markdown.
    md_path = ARTICLES_DIR / f"{filename}.md"
    md_path.write_text(
        markdown_document,
        encoding="utf-8",
    )

    # Save cleaned HTML.
    html_path = ARTICLES_DIR / f"{filename}.html"
    html_path.write_text(
        str(content),
        encoding="utf-8",
    )

    return {
        **article,
        "markdown_file": str(md_path),
        "html_file": str(html_path),
        "status": "success",
        "scraped_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# --------------------------------------------------
# Main execution
# --------------------------------------------------

def main():

    logger.info("Starting HMU administrative guide scraper")

    print("Fetching root webpage...")

    root_soup = fetch_page(ROOT_URL)

    # Extract the article links from the navigation.
    articles = extract_article_links(root_soup)

    print(f"Found {len(articles)} articles.")

    if not articles:
        raise RuntimeError(
            "No articles found. Check the navigation structure."
        )
    # print('root_soup: ', root_soup)
    # # Save the root page as well.
    # root_content = extract_main_content(root_soup)

    root_markdown = f"""---
title: "Οδηγός Διοικητικών Διαδικασιών"
source: "{ROOT_URL}"
scraped_at: "{datetime.now(timezone.utc).isoformat()}"
---

# Οδηγός Διοικητικών Διαδικασιών

## Περιεχόμενα

"""
    for article in articles:
        root_markdown += (
            f"- [{article['title']}]({article['url']})\n"
        )

    (OUTPUT_DIR / "root.md").write_text(
        root_markdown,
        encoding="utf-8",
    )

    # Preserve the original root HTML for reference.
    (OUTPUT_DIR / "root.html").write_text(
        str(root_soup),
        encoding="utf-8",
    )

    print("\nDiscovered article links:")

    for article in articles:
        print(
            f"{article['number']:02d}: "
            f"{article['title']} -> {article['url']}"
        )

    manifest = {
        "source": ROOT_URL,
        "scraped_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_articles": len(articles),
        "articles": [],
    }

    # Visit every article.
    for index, article in enumerate(articles, start=1):

        print(
            f"[{index}/{len(articles)}] "
            f"{article['title']}"
        )

        try:
            result = scrape_article(article)
            manifest["articles"].append(result)

            print("  Saved successfully.")

        except Exception as error:

            logger.exception(
                "Failed to scrape %s",
                article["url"],
            )

            manifest["articles"].append({
                **article,
                "status": "failed",
                "error": str(error),
            })

            print(f"  ERROR: {error}")

        time.sleep(DELAY_SECONDS)

    # Save manifest.
    manifest_path = OUTPUT_DIR / "manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    successful = sum(
        a["status"] == "success"
        for a in manifest["articles"]
    )

    print("\nScraping completed.")
    print(f"Total articles: {len(articles)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(articles) - successful}")
    print(f"Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()