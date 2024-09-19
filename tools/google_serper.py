import os
import sys
import json
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
import requests
from typing import Dict, Any
from config.load_configs import load_config

def format_results(organic_results: str) -> str:
    result_strings = []
    for result in organic_results:
        title = result.get('title', 'No Title')
        link = result.get('link', '#')
        snippet = result.get('snippet', 'No snippet available.')
        result_strings.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n---")
    
    return '\n'.join(result_strings)

def format_shopping_results(shopping_results: list) -> str:
    result_strings = []
    for result in shopping_results:
        title = result.get('title', 'No Title')
        link = result.get('link', '#')
        price = result.get('price', 'Price not available')
        source = result.get('source', 'Source not available')
        rating = result.get('rating', 'No rating')
        rating_count = result.get('ratingCount', 'No rating count')
        delivery = result.get('delivery', 'Delivery information not available')
        
        result_strings.append(f"Title: {title}\nSource: {source}\nPrice: {price}\nRating: {rating} ({rating_count} reviews)\nDelivery: {delivery}\nLink: {link}\n---")
    
    return '\n'.join(result_strings)

def serper_search(query: str, location: str) -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    load_config(config_path)
    search_url = "https://google.serper.dev/search"
    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': os.environ['SERPER_API_KEY']  # Make sure to set this environment variable
    }
    payload = json.dumps({"q": query, "gl": location})
    
    try:
        response = requests.post(search_url, headers=headers, data=payload)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4XX, 5XX)
        results = response.json()
        
        if 'organic' in results:
            # Return the raw results
            return {'organic_results': results['organic']}
        else:
            return {'organic_results': []}

    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.RequestException as req_err:
        return f"Request error occurred: {req_err}"
    except KeyError as key_err:
        return f"Key error occurred: {key_err}"
    except json.JSONDecodeError as json_err:
        return f"JSON decoding error occurred: {json_err}"
    
def serper_shopping_search(query: str, location: str) -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    load_config(config_path)
    search_url = "https://google.serper.dev/shopping"
    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': os.environ['SERPER_API_KEY']
    }
    payload = json.dumps({"q": query, "gl": location})
    
    try:
        response = requests.post(search_url, headers=headers, data=payload)
        response.raise_for_status()
        results = response.json()
        
        if 'shopping' in results:
            # Return the raw results
            return {'shopping_results': results['shopping']}
        else:
            return {'shopping_results': []}

    except requests.exceptions.RequestException as req_err:
        return f"Request error occurred: {req_err}"
    except json.JSONDecodeError as json_err:
        return f"JSON decoding error occurred: {json_err}"

def serper_scholar_search(query: str, location: str) -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    load_config(config_path)
    search_url = "https://google.serper.dev/scholar"
    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': os.environ['SERPER_API_KEY']  # Ensure this environment variable is set
    }
    payload = json.dumps({"q": query, "gl": location})
    
    try:
        response = requests.post(search_url, headers=headers, data=payload)
        response.raise_for_status()
        results = response.json()
        
        if 'organic' in results:
            # Return the raw results
            return {'scholar_results': results['organic']}
        else:
            return {'scholar_results': []}
    
    except requests.exceptions.RequestException as req_err:
        return f"Request error occurred: {req_err}"
    except json.JSONDecodeError as json_err:
        return f"JSON decoding error occurred: {json_err}"

# Example usage
if __name__ == "__main__":
    search_query = "NVIDIA RTX 6000"
    results = serper_search(search_query)
    print(results)