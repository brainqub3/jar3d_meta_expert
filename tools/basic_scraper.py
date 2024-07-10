import requests
from bs4 import BeautifulSoup

def scrape_website(url: str) -> dict:
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text content
        texts = soup.stripped_strings
        content = ' '.join(texts)
        
        # Limit the content to 4000 characters
        content = content[:8000]
        
        # Return the result as a dictionary
        return {
            "source": url,
            "content": content
        }
    
    except requests.RequestException as e:
        # Handle any requests-related errors
        return {
            "source": url,
            "content": f"Error scraping website: {str(e)}"
        }

# Example usage:
# result = scrape_website("https://example.com")
# print(result)