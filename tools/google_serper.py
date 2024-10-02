import os
import sys
import json
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
import requests
from typing import Dict, Any
from config.load_configs import load_config

# def format_results(organic_results: str) -> str:
#     result_strings = []
#     for result in organic_results:
#         title = result.get('title', 'No Title')
#         link = result.get('link', '#')
#         result_strings.append(f"Title: {title}\nLink: {link}---")
    
#     return '\n'.join(result_strings)

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
        'X-API-KEY': os.environ['SERPER_API_KEY']  # Ensure this environment variable is set
    }
    payload = json.dumps({"q": query, "gl": location})
    
    try:
        response = requests.post(search_url, headers=headers, data=payload)
        response.raise_for_status()
        results = response.json()
        
        simplified_results = []
        if results.get('organic') and isinstance(results['organic'], list):
            for idx, result in enumerate(results['organic']):
                if isinstance(result, dict):
                    title = result.get('title', 'No Title')
                    link = result.get('link', '#')
                    sitelinks = result.get('sitelinks', [])
                    # Extract sitelinks if they exist
                    if isinstance(sitelinks, list):
                        sitelinks = [{'title': s.get('title', ''), 'link': s.get('link', '')} for s in sitelinks]
                    else:
                        sitelinks = []
                    simplified_results.append({
                        'query': query,
                        'title': title,
                        'link': link,
                        'sitelinks': sitelinks
                    })
                else:
                    # Log or handle unexpected entry type
                    print(f"Entry at index {idx} in results['organic'] is not a dict: {type(result)}")
        else:
            print("No 'organic' results found or 'organic' is not a list.")
        
        return {'organic_results': simplified_results}

    except requests.exceptions.HTTPError as http_err:
        return {'error': f"HTTP error occurred: {http_err}"}
    except requests.exceptions.RequestException as req_err:
        return {'error': f"Request error occurred: {req_err}"}
    except KeyError as key_err:
        return {'error': f"Key error occurred: {key_err}"}
    except json.JSONDecodeError as json_err:
        return {'error': f"JSON decoding error occurred: {json_err}"}
    except Exception as ex:
        return {'error': str(ex)}

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

def format_search_results(search_results: Dict[str, Any]) -> str:
    """
    Formats the search results dictionary into a readable string.

    Args:
        search_results (Dict[str, Any]): The dictionary containing search results.

    Returns:
        str: A formatted string with the query, title, link, and sitelinks.
    """
    formatted_strings = []
    organic_results = search_results.get('organic_results', [])

    for result in organic_results:
        query = result.get('query', 'No Query')
        title = result.get('title', 'No Title')
        link = result.get('link', 'No Link')

        # Start formatting the result
        result_string = f"Query: {query}\nTitle: {title}\nLink: {link}"

        # Handle sitelinks if they exist
        sitelinks = result.get('sitelinks', [])
        if sitelinks:
            sitelinks_strings = []
            for sitelink in sitelinks:
                sitelink_title = sitelink.get('title', 'No Title')
                sitelink_link = sitelink.get('link', 'No Link')
                sitelinks_strings.append(f"    - {sitelink_title}: {sitelink_link}")
            sitelinks_formatted = "\nSitelinks:\n" + "\n".join(sitelinks_strings)
            result_string += sitelinks_formatted
        else:
            result_string += "\nSitelinks: None"

        # Add a separator between results
        formatted_strings.append(result_string + "\n" + "-" * 40)

    # Combine all formatted results into one stringfo
    final_string = "\n".join(formatted_strings)
    return final_string

# Example usage
if __name__ == "__main__":
    search_query = "NVIDIA RTX 6000"
    results = serper_search(search_query, "us")
    formatted_results = format_search_results(results)
    print(formatted_results)