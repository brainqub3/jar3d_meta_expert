# import requests
# from bs4 import BeautifulSoup

# def scrape_website(url: str) -> dict:
#     try:
#         # Send a GET request to the URL
#         response = requests.get(url)
#         response.raise_for_status()  # Raise an exception for bad status codes
        
#         # Parse the HTML content
#         soup = BeautifulSoup(response.content, 'html.parser')
        
#         # Extract text content
#         texts = soup.stripped_strings
#         content = ' '.join(texts)
        
#         # Limit the content to 4000 characters
#         content = content[:8000]
        
#         # Return the result as a dictionary
#         return {
#             "source": url,
#             "content": content
#         }
    
#     except requests.RequestException as e:
#         # Handle any requests-related errors
#         return {
#             "source": url,
#             "content": f"Error scraping website: {str(e)}"
#         }

# # Example usage:
# # result = scrape_website("https://example.com")
# # print(result)

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import random
from requests.exceptions import RequestException
from fake_useragent import UserAgent

class AdvancedWebScraper:
    def __init__(self, max_retries=3, backoff_factor=0.3, timeout=10):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.session = requests.Session()
        self.ua = UserAgent()

    def get_random_user_agent(self):
        return self.ua.random

    def scrape_website(self, url: str) -> dict:
        headers = {'User-Agent': self.get_random_user_agent()}
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text(separator=' ', strip=True)
                
                # Basic content cleaning
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                # Limit content length
                content = text[:8000]
                
                # Extract title
                title = soup.title.string if soup.title else "No title found"
                
                # Extract meta description
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                description = meta_desc['content'] if meta_desc else "No description found"
                
                # Extract links
                links = [urljoin(url, a.get('href')) for a in soup.find_all('a', href=True)]
                
                return {
                    "source": url,
                    "title": title,
                    "description": description,
                    "content": content,
                    "Potentially useful links": links[:10]  # Limit to first 10 links
                }
            
            except RequestException as e:
                if attempt == self.max_retries - 1:
                    return {
                        "source": url,
                        "error": f"Failed to scrape website after {self.max_retries} attempts: {str(e)}"
                    }
                else:
                    time.sleep(self.backoff_factor * (2 ** attempt))
                    continue

# Example usage:
# scraper = AdvancedWebScraper()
# result = scraper.scrape_website("https://example.com")
# print(result)