import sys
import os
import io
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
import concurrent.futures
import functools
import requests
import numpy as np
import faiss
import traceback
import tempfile
from typing import Dict, List
from termcolor import colored
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from flashrank import Ranker, RerankRequest
from llmsherpa.readers import LayoutPDFReader
from langchain.schema import Document
from config.load_configs import load_config
from langchain_community.docstore.in_memory import InMemoryDocstore
from fake_useragent import UserAgent
from multiprocessing import Pool, cpu_count

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
load_config(config_path)

ua = UserAgent()
os.environ["USER_AGENT"] = ua.random
os.environ["FAISS_OPT_LEVEL"] = "generic"

def timeout(max_timeout):
    """Timeout decorator, parameter in seconds."""
    def timeout_decorator(item):
        """Wrap the original function."""
        @functools.wraps(item)
        def func_wrapper(*args, **kwargs):
            """Closure for function."""
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(item, *args, **kwargs)
                try:
                    return future.result(max_timeout)
                except concurrent.futures.TimeoutError:
                    return [Document(page_content=f"Timeout occurred while processing URL: {args[0]}", metadata={"source": args[0]})]
        return func_wrapper
    return timeout_decorator


@timeout(20)  # 20 second timeout
def intelligent_chunking(url: str) -> List[Document]:
    try:
        print(colored(f"\n\nStarting Intelligent Chunking with LLM Sherpa for URL: {url}\n\n", "green"))
        llmsherpa_api_url = os.environ.get('LLM_SHERPA_SERVER')

        if not llmsherpa_api_url:
            raise ValueError("LLM_SHERPA_SERVER environment variable is not set")
        
        corpus = []

        try: 
            print(colored("Starting LLM Sherpa LayoutPDFReader...\n\n", "yellow"))
            reader = LayoutPDFReader(llmsherpa_api_url)
            doc = reader.read_pdf(url)
            print(colored("Finished LLM Sherpa LayoutPDFReader...\n\n", "yellow"))
        except Exception as e:
            print(colored(f"Error in LLM Sherpa LayoutPDFReader: {str(e)}", "red"))
            traceback.print_exc()
            doc = None
        
        if doc:
            for chunk in doc.chunks():
                document = Document(
                    page_content=chunk.to_context_text(),
                    metadata={"source": url}
                )
                corpus.append(document)
            
            print(colored(f"Created corpus with {len(corpus)} documents", "green"))
        
        if not doc:
            print(colored(f"No document to append to corpus", "red"))
        
        return corpus
    
    except concurrent.futures.TimeoutError:
        print(colored(f"Timeout occurred while processing URL: {url}", "red"))
        return [Document(page_content=f"Timeout occurred while processing URL: {url}", metadata={"source": url})]
    except Exception as e:        
        print(colored(f"Error in Intelligent Chunking for URL {url}: {str(e)}", "red"))
        traceback.print_exc()
        return [Document(page_content=f"Error in Intelligent Chunking for URL: {url}", metadata={"source": url})]


def index_and_rank(corpus: List[Document], query: str, top_percent: float = 60, batch_size: int = 25) -> List[Dict[str, str]]:
    print(colored(f"\n\nStarting indexing and ranking with FastEmbeddings and FAISS for {len(corpus)} documents\n\n", "green"))
    embeddings = FastEmbedEmbeddings(model_name='jinaai/jina-embeddings-v2-small-en', max_length=512)

    print(colored("\n\nCreating FAISS index...\n\n", "green"))

    try:
        # Initialize an empty FAISS index
        index = None
        docstore = InMemoryDocstore({})
        index_to_docstore_id = {}

        # Process documents in batches
        for i in range(0, len(corpus), batch_size):
            batch = corpus[i:i+batch_size]
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]

            print(f"Processing batch {i // batch_size + 1} with {len(texts)} documents")

            # Embed the batch
            batch_embeddings = embeddings.embed_documents(texts)

            # Convert embeddings to numpy array with float32 dtype
            batch_embeddings_np = np.array(batch_embeddings, dtype=np.float32)

            if index is None:
                # Create the index with the first batch
                index = faiss.IndexFlatIP(batch_embeddings_np.shape[1])
            
            # Normalize the embeddings
            faiss.normalize_L2(batch_embeddings_np)

            # Add embeddings to the index
            start_id = len(index_to_docstore_id)
            index.add(batch_embeddings_np)
            
            # Update docstore and index_to_docstore_id
            for j, (text, metadata) in enumerate(zip(texts, metadatas)):
                doc_id = f"{start_id + j}"
                docstore.add({doc_id: Document(page_content=text, metadata=metadata)})
                index_to_docstore_id[start_id + j] = doc_id

        print(f"Total documents indexed: {len(index_to_docstore_id)}")

        # Create a FAISS retriever
        retriever = FAISS(embeddings, index, docstore, index_to_docstore_id)

        # Perform the search
        k = min(40, len(corpus))  # Ensure we don't try to retrieve more documents than we have
        docs = retriever.similarity_search_with_score(query, k=k)
        print(colored(f"\n\nRetrieved {len(docs)} documents\n\n", "green"))
        
        passages = []
        for idx, (doc, score) in enumerate(docs, start=1):
            try:
                passage = {
                    "id": idx,
                    "text": doc.page_content,
                    "meta": doc.metadata,
                    "score": float(score)  # Convert score to float
                }
                passages.append(passage)
            except Exception as e:
                print(colored(f"Error in creating passage: {str(e)}", "red"))
                traceback.print_exc()

        print(colored("\n\nRe-ranking documents...\n\n", "green"))
        ranker = Ranker(cache_dir=tempfile.mkdtemp())
        rerankrequest = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerankrequest)
        print(colored("\n\nRe-ranking complete\n\n", "green"))

        # Sort results by score in descending order
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

        # Calculate the number of results to return based on the percentage
        num_results = max(1, int(len(sorted_results) * (top_percent / 100)))
        top_results = sorted_results[:num_results]

        final_results = [
            {
                "text": result['text'],
                "meta": result['meta'],
                "score": result['score']
            }
            for result in top_results
        ]

        print(colored(f"\n\nReturned top {top_percent}% of results ({len(final_results)} documents)\n\n", "green"))

        # Add debug information about scores
        scores = [result['score'] for result in results]
        print(f"Score distribution: min={min(scores):.4f}, max={max(scores):.4f}, mean={np.mean(scores):.4f}, median={np.median(scores):.4f}")
        print(f"Unique scores: {len(set(scores))}")
        if final_results:
            print(f"Score range for top {top_percent}% results: {final_results[-1]['score']:.4f} to {final_results[0]['score']:.4f}")

    except Exception as e:
        print(colored(f"Error in indexing and ranking: {str(e)}", "red"))
        traceback.print_exc()
        final_results = [{"text": "Error in indexing and ranking", "meta": {"source": "unknown"}, "score": 0.0}]

    return final_results

def run_rag(urls: List[str], query: str) -> List[Dict[str, str]]:
    # Use ThreadPoolExecutor instead of multiprocessing
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(urls), 3)) as executor:
        futures = [executor.submit(intelligent_chunking, url) for url in urls]
        chunks_list = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # Flatten the list of lists into a single corpus
    corpus = [chunk for chunks in chunks_list for chunk in chunks]
    print(colored(f"\n\nTotal documents in corpus after chunking: {len(corpus)}\n\n", "green"))
    
    ranked_docs = index_and_rank(corpus, query)
    return ranked_docs

# def run_rag(urls: List[str], query: str) -> List[Dict[str, str]]:
#     # Use multiprocessing to chunk URLs in parallel
#     with Pool(processes=min(cpu_count(), len(urls))) as pool:
#         chunks_list = pool.map(intelligent_chunking, urls)
    
#     # Flatten the list of lists into a single corpus
#     corpus = [chunk for chunks in chunks_list for chunk in chunks]
    
#     print(colored(f"\n\nTotal documents in corpus after chunking: {len(corpus)}\n\n", "green"))
    
#     ranked_docs = index_and_rank(corpus, query)
#     return ranked_docs

if __name__ == "__main__":
    # For testing purposes.
    url1 = "https://www.amazon.com/dp/B0CX23GFMJ/ref=fs_a_mbt2_us4"
    url2 = "https://www.amazon.com/dp/B0CX23V2ZK/ref=fs_a_mbt2_us3"
    url3 = "https://der8auer.com/x570-motherboard-vrm-overview/"

    query = "cheapest macbook"

    urls = [url1, url2, url3]
    results = run_rag(urls, query)

    print(f"\n\n RESULTS: {results}")