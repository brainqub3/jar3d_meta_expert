from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

def scrape(url):
    loader = AsyncChromiumLoader([url])
    html = loader.load() 
    # Transform
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["p"])
    print(docs_transformed[0].page_content)

if __name__ == "__main__":
    scrape("https://arxiv.org/pdf/2401.12954")