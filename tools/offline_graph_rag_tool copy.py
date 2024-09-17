import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
import concurrent.futures
import functools
import numpy as np
import faiss
import traceback
import tempfile
from typing import Dict, List
from termcolor import colored
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
# from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from flashrank import Ranker, RerankRequest
from llmsherpa.readers import LayoutPDFReader
from langchain.schema import Document
from config.load_configs import load_config
from langchain_community.docstore.in_memory import InMemoryDocstore
from fake_useragent import UserAgent

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


# Change: Added function to deduplicate re-ranked results.
def deduplicate_results(results, rerank=True):
    seen = set()
    unique_results = []
    for result in results:
        # Create a tuple of the content and source to use as a unique identifier
        if rerank:
            identifier = (result['text'], result['meta'])
        else:
            # When not reranking, result is a tuple (doc, score)
            doc, score = result
            identifier = (doc.page_content, doc.metadata.get('source', ''))
        if identifier not in seen:
            seen.add(identifier)
            unique_results.append(result)
    return unique_results


def index_and_rank(corpus: List[Document], query: str, top_percent: float = 20, batch_size: int = 25) -> List[Dict[str, str]]:
    print(colored(f"\n\nStarting indexing and ranking with FastEmbeddings and FAISS for {len(corpus)} documents\n\n", "green"))
    CACHE_DIR = "/app/fastembed_cache"
    embeddings = FastEmbedEmbeddings(model_name='jinaai/jina-embeddings-v2-small-en', max_length=512, cache_dir=CACHE_DIR)

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

        # Change: Retrieve documents based on query in metadata  
        similarity_cache = {}
        docs = []
        for doc in corpus:
            query = doc.metadata.get('query', '')
            # Check if we've already performed this search
            if query in similarity_cache:
                cached_results = similarity_cache[query]
                docs.extend(cached_results)
            else:
                # Perform the similarity search
                search_results = retriever.similarity_search_with_score(query, k=k)
                
                # Cache the results
                similarity_cache[query] = search_results
                
                # Add to docs
                docs.extend(search_results)

        docs = deduplicate_results(docs, rerank=False)

        print(colored(f"\n\nRetrieved {len(docs)} documents\n\n", "green"))
        
        passages = []
        for idx, (doc, score) in enumerate(docs, start=1):
            try:
                passage = {
                    "id": idx,
                    "text": doc.page_content,
                    "meta": doc.metadata.get("source", {"source": "unknown"}),
                    "score": float(score)  # Convert score to float
                }
                passages.append(passage)
            except Exception as e:
                print(colored(f"Error in creating passage: {str(e)}", "red"))
                traceback.print_exc()

        print(colored("\n\nRe-ranking documents...\n\n", "green"))
        # Change: reranker done based on query in metadata
        CACHE_DIR_RANKER = "/app/reranker_cache"
        ranker = Ranker(cache_dir=CACHE_DIR_RANKER)
        results = []
        processed_queries = set()

        # Perform reranking with query caching
        for doc in corpus:
            query = doc.metadata.get('query', '')
            
            # Skip if we've already processed this query
            if query in processed_queries:
                continue
            
            rerankrequest = RerankRequest(query=query, passages=passages)
            result = ranker.rerank(rerankrequest)
            results.extend(result)
            
            # Mark this query as processed
            processed_queries.add(query)

        results = deduplicate_results(results, rerank=True)

        print(colored(f"\n\nRe-ranking complete with {len(results)} documents\n\n", "green"))

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

def run_hybrid_graph_retrrieval(graph: Neo4jGraph = None, corpus: List[Document] = None, query: str = None, hybrid: bool = False):
    print(colored(f"\n\Initiating Retrieval...\n\n", "green"))

    if hybrid:
        print(colored("Running Hybrid Retrieval...", "yellow"))
        unstructured_data = index_and_rank(corpus, query)

        query = f"""
        MATCH p = (n)-[r]->(m)
        WHERE COUNT {{(n)--()}} > 30
        RETURN p AS Path
        LIMIT 85
        """
        response = graph.query(query)
        retrieved_context = f"Important Relationships:{response}\n\n Additional Context:{unstructured_data}"

    else:
        print(colored("Running Dense Only Retrieval...", "yellow"))
        unstructured_data = index_and_rank(corpus, query)
        retrieved_context = f"Additional Context:{unstructured_data}"

    return retrieved_context


@timeout(20)  # Change: Takes url and query as input
def intelligent_chunking(url: str, query: str) -> List[Document]:
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
                    metadata={"source": url, "query": query} # Change: Added query to metadata
                )

                if len(document.page_content) > 30:
                    corpus.append(document)
            
            print(colored(f"Created corpus with {len(corpus)} documents", "green"))
            
        
        if not doc:
            print(colored(f"No document to append to corpus", "red"))
        
        # print(colored(f"DEBUG: Corpus: {corpus}", "yellow"))
        return corpus
    
    except concurrent.futures.TimeoutError:
        print(colored(f"Timeout occurred while processing URL: {url}", "red"))
        return [Document(page_content=f"Timeout occurred while processing URL: {url}", metadata={"source": url})]
    except Exception as e:        
        print(colored(f"Error in Intelligent Chunking for URL {url}: {str(e)}", "red"))
        traceback.print_exc()
        return [Document(page_content=f"Error in Intelligent Chunking for URL: {url}", metadata={"source": url})]


def clear_neo4j_database(graph: Neo4jGraph):
    """
    Clear all nodes and relationships from the Neo4j database.
    """
    try:
        print(colored("\n\nClearing Neo4j database...\n\n", "yellow"))
        # Delete all relationships first
        graph.query("MATCH ()-[r]->() DELETE r")
        # Then delete all nodes
        graph.query("MATCH (n) DELETE n")
        print(colored("Neo4j database cleared successfully.\n\n", "green"))
    except Exception as e:
        print(colored(f"Error clearing Neo4j database: {str(e)}", "red"))
        traceback.print_exc()

def process_document(doc: Document, llm_transformer: LLMGraphTransformer, doc_num: int, total_docs: int) -> List:
    print(colored(f"\n\nStarting Document {doc_num} of {total_docs}: {doc.page_content[:100]}\n\n", "yellow"))
    graph_document = llm_transformer.convert_to_graph_documents([doc])
    print(colored(f"\nFinished Document {doc_num}\n\n", "green"))
    return graph_document

def create_graph_index(
    documents: List[Document] = None, 
    allowed_relationships: list[str] = None, 
    allowed_nodes: list[str] = None, 
    query: str = None, 
    graph: Neo4jGraph = None,
    max_threads: int = 5
) -> Neo4jGraph:
    
    if os.environ.get('LLM_SERVER') == "openai":
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

    else:
        llm = ChatAnthropic(temperature=0, model_name="claude-3-haiku-20240307")

    # llm = ChatAnthropic(temperature=0, model_name="claude-3-haiku-20240307")

    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
        node_properties=True,
        relationship_properties=True
    )

    graph_documents = []
    total_docs = len(documents)

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Create a list of futures
        futures = [
            executor.submit(process_document, doc, llm_transformer, i+1, total_docs)
            for i, doc in enumerate(documents)
        ]

        # Process completed futures
        for future in concurrent.futures.as_completed(futures):
            graph_documents.extend(future.result())

    print(colored(f"\n\nTotal graph documents: {len(graph_documents)}", "green"))
    # print(colored(f"\n\DEBUG graph documents: {graph_documents}", "red"))

    graph_documents = [graph_documents]
    flattened_graph_list = [item for sublist in graph_documents for item in sublist]
    # print(colored(f"\n\DEBUG Flattened graph documents: {flattened_graph_list}", "yellow"))


    graph.add_graph_documents(
        flattened_graph_list, 
        baseEntityLabel=True, 
        include_source=True,
    )

    return graph


def run_rag(urls: List[str], allowed_nodes: list[str] = None, allowed_relationships: list[str] = None, query: List[str] = None, hybrid: bool = False) -> List[Dict[str, str]]:
    # Change: adapted to take query and url as input.
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(urls), 5)) as executor:  
            futures = [executor.submit(intelligent_chunking, url, query) for url, query in zip(urls, query)]
            chunks_list = [future.result() for future in concurrent.futures.as_completed(futures)]
    

    corpus = [item for sublist in chunks_list for item in sublist]

    print(colored(f"\n\nTotal documents in corpus after chunking: {len(corpus)}\n\n", "green"))


    print(colored(f"\n\n DEBUG HYBRID VALUE: {hybrid}\n\n", "yellow"))
    
    if hybrid:
        print(colored(f"\n\n Creating Graph Index...\n\n", "green"))
        graph = Neo4jGraph()
        clear_neo4j_database(graph)
        graph = create_graph_index(documents=corpus, allowed_nodes=allowed_nodes, allowed_relationships=allowed_relationships, query=query, graph=graph)
    else:
        graph = None

    retrieved_context = run_hybrid_graph_retrrieval(graph=graph, corpus=corpus, query=query, hybrid=hybrid)

    retrieved_context = str(retrieved_context)

    return retrieved_context

# if __name__ == "__main__":
#     # For testing purposes.
#     url1 = "https://www.reddit.com/r/microsoft/comments/1bkikl1/regretting_buying_copilot_for_microsoft_365"
#     url2 = "'https://www.reddit.com/r/microsoft_365_copilot/comments/1chtqtg/do_you_actually_find_365_copilot_useful_in_your"
#     # url3 = "https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/"

#     # query = "cheapest macbook"

#     # urls = [url1, url2, url3]
#     urls = [url1, url2]
#     query = ["Co-pilot Microsoft"]
#     allowed_nodes = None
#     allowed_relationships = None
#     hybrid = False
#     results = run_rag(urls, allowed_nodes=allowed_nodes, allowed_relationships=allowed_relationships, query=query, hybrid=hybrid)

#     print(colored(f"\n\n RESULTS: {results}", "green"))

#     print(f"\n\n RESULTS: {results}")