from typing import Dict, Any, TypeVar
from termcolor import colored
from .agent_base import ToolCallingAgent, StateT
from tools.google_serper import serper_search, format_search_results
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

class SerperDevAgent(ToolCallingAgent[StateT]):
    """
    # Functionality:
    This agent performs Google web searches based on a list of queries you provide. It returns a formatted list of organic search results, including the query, title, link, and sitelinks for each result.

    ## Inputs:
    - **queries**: A list of search query strings.
    - **location**: Geographic location code for the search (e.g., 'us', 'gb', 'nl', 'ca'). Defaults to 'us'.

    ## Outputs:
    - A formatted string representing the organic search engine results page (SERP), including:
        - Query
        - Title
        - Link
        - Sitelinks

    ## When to Use:
    - When you need to retrieve search engine results for specific queries.
    - When you require URLs from search results for further investigation.

    ## Important Notes:
    - This tool **only** provides search result summaries; it does **not** access or retrieve content from the linked web pages.
    - To obtain detailed content or specific information from the web pages listed in the search results, you should use the **WebScraperAgent** or the **OfflineRAGWebsearchAgent** with the URLs obtained from this tool.

    ## Example Workflow:
    1. **Search**: Use this agent with queries like `["latest advancements in AI"]`.
    2. **Retrieve URLs**: Extract the list of URLs from the search results.
    3. **Deep Dive**:
        - Use web scraping with the extracted URLs to get the full content of the pages.
        - Use RAG to extract specific data from web pages.

    # Remember
    You should provide the inputs as suggested.

    --------------------------------
    """

    def __init__(self, name: str, model: str = "claude-v1", server: str = "claude", temperature: float = 0):
        super().__init__(name, model=model, server=server, temperature=temperature)
        self.location = "us"  # Default location for search
        print(f"SerperDevAgent '{self.name}' initialized.")

    def get_guided_json(self, state: StateT = None) -> Dict[str, Any]:
        """
        Define the guided JSON schema expecting a list of search queries.
        """
        guided_json_schema = {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A search query string."
                    },
                    "description": "A list of search query strings."
                },
                "location": {
                    "type": "string",
                    "description": (
                        "The geographic location for the search results. "
                        "Available locations: 'us' (United States), 'gb' (United Kingdom), "
                        "'nl' (The Netherlands), 'ca' (Canada)."
                    )
                }
            },
            "required": ["queries", "location"],
            "additionalProperties": False
        }
        return guided_json_schema

    def execute_tool(self, tool_response: Dict[str, Any], state: StateT = None) -> Any:
        """
        Execute the search tool using the provided tool response, handling multiple queries concurrently.
        Returns the search results as a concatenated string.
        """
        queries = tool_response.get("queries")
        location = tool_response.get("location", self.location)
        if not queries:
            raise ValueError("Search queries are missing from the tool response")
        print(f"{self.name} is searching for queries: {queries} in location: {location}")

        # Define a function for searching a single query
        def search_query(query):
            print(f"Searching for '{query}' in location '{location}'")
            result = serper_search(query, location)
            formatted_result_str = format_search_results(result)
            print(f"Obtained search results for query: '{query}'")
            return formatted_result_str  # Return only the formatted result string

        # Collect all formatted result strings
        search_results_list = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_query = {executor.submit(search_query, query): query for query in queries}
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    result = future.result()
                    search_results_list.append(result)  # Append the result string directly
                except Exception as exc:
                    print(f"Exception occurred while searching for query '{query}': {exc}")
                    error_message = f"Error for query '{query}': {exc}"
                    search_results_list.append(error_message)

        # Combine all search results into a single string
        combined_results = "\n".join(search_results_list)
        print(colored(f"DEBUG: {self.name} search results: {combined_results} \n\n Type:{type(combined_results)}", "green"))

        # Return the combined search results as a string
        return combined_results

if __name__ == "__main__":
    # Create an instance of SerperDevAgent for testing
    agent = SerperDevAgent("TestSerperAgent")

    # Create a sample tool response
    test_tool_response = {
        "queries": ["Python programming", "Machine learning basics"],
        "location": "us"
    }

    # Create a sample state (can be None or an empty dict for this test)
    test_state = {}

    # Execute the tool and print the results
    try:
        results = agent.execute_tool(test_tool_response, test_state)
        print("Search Results:")
        print(results)
    except Exception as e:
        print(f"An error occurred: {e}")

    # You can add more test cases or assertions here to verify the functionality