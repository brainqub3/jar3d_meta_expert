import os
from termcolor import colored
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage
from fake_useragent import UserAgent

ua = UserAgent()
os.environ["USER_AGENT"] = ua.random

def scraper(url: str) -> dict:
    print(colored(f"\n\nStarting basic scraping with URL: {url}\n\n", "green"))
    try:
        print(colored(f"Starting HTML scraper with URL: {url}", "green"))
        loader = AsyncChromiumLoader([url])
        html = loader.load()
        # Transform
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["p"])
        # Combine content from all paragraphs
        content = "\n".join([doc.page_content for doc in docs_transformed])
        result = {"source": url, "content": content}
        print(result)
        return result
    except Exception as e:
        try:
            print(colored(f"Starting PDF scraper with URL: {url}", "green"))
            loader = PyPDFLoader(url)
            pages = loader.load_and_split()
            # Combine content from all pages
            content = "\n".join([page.page_content for page in pages])
            result = {"source": url, "content": content}
            print(result)
            return result
        except Exception as e:
            result = {
                "source": url,
                "content": "Unsupported document type, supported types are 'html' and 'pdf'."
            }
            print(result)
            return result

def scrape_urls(urls: list) -> list:
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(scraper, url): url for url in urls}
        for future in as_completed(future_to_url):
            try:
                data = future.result()
                results.append(data)
            except Exception as exc:
                url = future_to_url[future]
                print(f"{url} generated an exception: {exc}")
    return results

if __name__ == "__main__":
    urls_to_scrape = [
        "https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/pdf/",
        "https://example.com",
        # Add more URLs as needed
    ]
    scrape_results = scrape_urls(urls_to_scrape)
    for result in scrape_results:
        print(result)