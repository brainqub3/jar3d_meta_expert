from typing import Dict, Any, TypeVar
from termcolor import colored
from .agent_base import ToolCallingAgent, StateT
from tools.google_serper import serper_shopping_search, format_shopping_results
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

class SerperShoppingAgent(ToolCallingAgent[StateT]):
    """
    # Functionality:
    This agent performs Google Shopping searches based on a list of queries you provide. It returns a formatted list of shopping results, including the title, price, source, rating, and link for each result.

    ## Inputs:
    - **queries**: A list of shopping query strings.
    - **location**: Geographic location code for the search (e.g., 'us', 'gb', 'nl', 'ca'). Defaults to 'us'.

    ## Outputs:
    - A formatted string representing the shopping search results, including:
        - Title
        - Source
        - Price
        - Rating
        - Delivery
        - Link

    ## When to Use:
    - When you need to retrieve product information from Google Shopping for specific queries.
    - When you require product details such as price, rating, and source.

    ## Important Notes:
    - This tool **only** provides shopping result summaries; it does **not** access or retrieve content from the linked web pages.
    - To obtain detailed content or specific information from the product pages, you should use the **WebScraperAgent** or the **OfflineRAGWebsearchAgent** with the URLs obtained from this tool.

    # Remember
    You should provide the inputs as suggested.

    --------------------------------
    """

    def __init__(self, name: str, model: str = "claude-v1", server: str = "claude", temperature: float = 0):
        super().__init__(name, model=model, server=server, temperature=temperature)
        self.location = "us"  # Default location for search
        print(f"SerperShoppingAgent '{self.name}' initialized.")

    def get_guided_json(self, state: StateT = None) -> Dict[str, Any]:
        """
        Define the guided JSON schema expecting a list of shopping queries.
        """
        guided_json_schema = {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A shopping query string."
                    },
                    "description": "A list of shopping query strings."
                },
                "location": {
                    "type": "string",
                    "description": (
                        "The geographic location for the shopping search results. "
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
        Execute the shopping search tool using the provided tool response, handling multiple queries concurrently.
        Returns the shopping results as a concatenated string.
        """
        queries = tool_response.get("queries")
        location = tool_response.get("location", self.location)
        if not queries:
            raise ValueError("Shopping queries are missing from the tool response")
        print(f"{self.name} is searching for shopping queries: {queries} in location: {location}")

        # Define a function for searching a single query
        def search_query(query):
            print(f"Searching for '{query}' in location '{location}'")
            result = serper_shopping_search(query, location)
            if 'error' in result:
                print(f"Error retrieving shopping results for query '{query}': {result['error']}")
                return f"Error retrieving shopping results for query '{query}': {result['error']}"
            formatted_result_str = format_shopping_results(result.get('shopping_results', []))
            print(f"Obtained shopping results for query: '{query}'")
            return formatted_result_str  # Return only the formatted result string

        # Collect all formatted result strings
        shopping_results_list = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_query = {executor.submit(search_query, query): query for query in queries}
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    result = future.result()
                    shopping_results_list.append(result)  # Append the result string directly
                except Exception as exc:
                    print(f"Exception occurred while searching for query '{query}': {exc}")
                    error_message = f"Error for query '{query}': {exc}"
                    shopping_results_list.append(error_message)

        # Combine all shopping results into a single string
        combined_results = "\n".join(shopping_results_list)
        print(colored(f"DEBUG: {self.name} shopping results: {combined_results} \n\n Type:{type(combined_results)}", "green"))

        # Return the combined shopping results as a string
        return combined_results

# if __name__ == "__main__":
#     # Create an instance of SerperShoppingAgent for testing
#     agent = SerperShoppingAgent("TestSerperShoppingAgent")

#     # Create a sample tool response
#     test_tool_response = {
#         "queries": ["Apple iPhone 14", "Samsung Galaxy S22"],
#         "location": "us"
#     }

#     # Create a sample state (can be None or an empty dict for this test)
#     test_state = {}

#     # Execute the tool and print the results
#     try:
#         results = agent.execute_tool(test_tool_response, test_state)
#         print("Shopping Results:")
#         print(results)
#     except Exception as e:
#         print(f"An error occurred: {e}")

#     # You can add more test cases or assertions here to verify the functionality
