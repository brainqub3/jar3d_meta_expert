# agents/intelligent_scraper_agent.py

from typing import Dict, Any, TypeVar, List
import json
from .agent_base import ToolCallingAgent, StateT
from tools.offline_graph_rag_tool import run_rag
from langsmith import traceable

class OfflineRAGWebsearchAgent(ToolCallingAgent[StateT]):
    """
    # Functionality:
    This agent extracts **specific information** from web pages by processing them using a Retrieval-Augmented Generation (RAG) approach. Use this tool when you need precise answers or data related to particular queries from web pages.

    ## Inputs:
        - **url**: The URL of the web page to process.
        - **query**: The specific question or topic to search for within the web page.

    You can provide multiple url-query pairs.

    ## Outputs:
    - A JSON-formatted string containing the specific information retrieved from each webpage, along with its corresponding URL.

    ## When to Use:
    - When you need to extract specific pieces of information or answers to particular questions from web pages.
    - After obtaining URLs and needing targeted information from those pages.

    ## Important Notes:
    - This tool processes the content of each URL to find information relevant to the provided query.
    - It is more efficient than web scraping when you only need specific data rather than the entire page content.

    ## Example Workflow:
    1. **Get URLs**: Get the search engine results page (SERP).
    2. **Define Queries**: Formulate specific queries for the information you need from each URL.
    3. **Extract Information**: Use this agent to retrieve targeted information.

    # Remember
    You should provide the inputs as suggested.

    --------------------------------
    """

    def __init__(self, name: str, model: str = "claude-3-5-sonnet-20240620", server: str = "anthropic", temperature: float = 0):
        super().__init__(name, model=model, server=server, temperature=temperature)
        print(f"OfflineRAGWebsearchAgent '{self.name}' initialized.")

    def get_guided_json(self, state: StateT = None) -> Dict[str, Any]:
        """
        Get guided JSON schema for the intelligent chunking tool, expecting a list of URL-query pairs.
        """
        guided_json_schema = {
            "type": "object",
            "properties": {
                "url_query_pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL of the web page to process."
                            },
                            "query": {
                                "type": "string",
                                "description": "The specific query or topic to search for on the web page."
                            }
                        },
                        "required": ["url", "query"],
                        "additionalProperties": False
                    },
                    "description": "A list of URL and query pairs to process.",
                }
            },
            "required": ["url_query_pairs"],
            "additionalProperties": False
        }
        return guided_json_schema

    def execute_tool(self, tool_response: Dict[str, Any], state: StateT = None) -> Any:
        """
        Execute the run_rag method on a list of URL-query pairs.
        Returns the results as a JSON-formatted string.
        """
        url_query_pairs = tool_response.get("url_query_pairs")
        if not url_query_pairs:
            raise ValueError("url_query_pairs are missing from the tool response")
        print(f"{self.name} is processing URL-query pairs: {url_query_pairs}")

        # Extract URLs and queries
        urls = [item["url"] for item in url_query_pairs]
        queries = [item["query"] for item in url_query_pairs]

        # Call the run_rag method from offline_graph_rag_tool
        results = run_rag(
            urls=urls,
            allowed_nodes=None,
            allowed_relationships=None,
            query=queries,
            rag_mode="Dense"  # or "Hybrid" based on your requirements
        )

        # Convert the results to JSON string
        results_str = json.dumps(results)
        return results_str

    # Remove the invoke method to rely on the base class implementation