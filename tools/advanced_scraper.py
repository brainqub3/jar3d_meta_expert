from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import PyPDFLoader


def scraper(url: str, doc_type: str) -> dict:
    if doc_type == "html":
        try:
            loader = AsyncChromiumLoader([url])
            html = loader.load() 
            # Transform
            bs_transformer = BeautifulSoupTransformer()
            docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["p"])
            return {"source":url, "content":docs_transformed[0].page_content}
        except Exception as e:
            return {"source": url, "content": f"Error scraping website: {str(e)}"}
    elif doc_type == "pdf":
        try:
            loader = PyPDFLoader(url)
            pages = loader.load_and_split()
            return {"source":url, "content":pages}
        except Exception as e:
            return {"source": url, "content": f"Error scraping PDF: {str(e)}"}
    else:
        return {"source": url, "content": "Unsupported document type, supported types are 'html' and 'pdf'."}

# def scrape_html(url):
#     loader = AsyncChromiumLoader([url])
#     html = loader.load() 
#     # Transform
#     bs_transformer = BeautifulSoupTransformer()
#     docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["p"])
#     print(docs_transformed[0].page_content)

# def scrape_pdf(url):
#     loader = PyPDFLoader(url)
#     pages = loader.load_and_split()
#     return pages


if __name__ == "__main__":
    scraper("https://arxiv.org/pdf/2401.12954")