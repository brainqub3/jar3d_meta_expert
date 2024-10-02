# agents/web_scraper_agent.py

from typing import Dict, Any, TypeVar
from concurrent.futures import ThreadPoolExecutor, as_completed
from .agent_base import ToolCallingAgent, StateT
from tools.basic_scraper import scraper
import json

class WebScraperAgent(ToolCallingAgent[StateT]):
    """
    # Functionality:
    This agent scrapes the **entire content** from web pages provided in a list of URLs. Use this tool when you need comprehensive information or global context from web pages.

    ## Inputs:
    - **urls**: A list of URLs to scrape.

    ## Outputs:
    - A JSON-formatted string containing the scraped content from each webpage, mapped to its corresponding URL.

    ## When to Use:
    - When you need to retrieve the full text content of web pages for analysis.
    - After obtaining URLs and requiring detailed content from those pages.

    ## Important Notes:
    - This tool retrieves **all available text content** from the specified URLs.
    - If you only need specific information from the web pages, consider using the `OfflineRAGWebsearchAgent` instead.

    ## Example Workflow:
    1. **Obtain URLs**: Get search results and extract URLs.
    2. **Scrape Content**: Use web scraping with the list of URLs to scrape full content.
    3. **Utilize Data**: Analyze or process the scraped content as needed.

    # Remember
    You should provide the inputs as suggested.

    --------------------------------
    """

    def __init__(self, name: str, model: str = "claude-3-5-sonnet-20240620", server: str = "anthropic", temperature: float = 0):
        super().__init__(name, model=model, server=server, temperature=temperature)
        print(f"WebScraperAgent '{self.name}' initialized.")

    def get_guided_json(self, state: StateT = None) -> Dict[str, Any]:
        """
        Get guided JSON schema for the scraper tool, expecting a list of URLs.
        """
        guided_json_schema = {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A valid URL to scrape."
                    },
                    "description": "A list of URLs to scrape.",
                }
            },
            "required": ["urls"],
            "additionalProperties": False
        }
        return guided_json_schema

    def execute_tool(self, tool_response: Dict[str, Any], state: StateT = None) -> Any:
        """
        Execute the scraper tool on a list of URLs concurrently using multi-threading.
        Returns the scrape results as a JSON-formatted string.
        """
        urls = tool_response.get("urls")
        if not urls:
            raise ValueError("URLs are missing from the tool response")
        print(f"{self.name} is scraping URLs: {urls}")
        
        # Define a function for scraping a single URL
        def scrape_url(url):
            print(f"{self.name} is scraping URL: {url}")
            scrape_result = scraper(url)
            print(f"{self.name} obtained scrape result for URL: {url}")
            return url, scrape_result

        # Use ThreadPoolExecutor to perform scraping concurrently
        scrape_results = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(scrape_url, url): url for url in urls}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    _, result = future.result()
                    scrape_results[url] = result
                except Exception as exc:
                    print(f"{self.name} generated an exception for URL {url}: {exc}")
                    scrape_results[url] = {"error": str(exc)}
        
        # Convert the scrape results to a JSON string
        scrape_results_str = json.dumps(scrape_results)
        # print(f"DEBUG: {self.name} scrape results: {scrape_results_str} \n\n Type:{type(scrape_results_str)}")
        
        # Return the scrape results as a JSON string
        return scrape_results_str

    # Remove the invoke method to rely on the base class implementation