import requests
import time
import yaml
import json
import os
import logging
from typing import List, Dict
from utils.logging import log_function, setup_logging
from config.load_configs import load_config

setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ClaudeModel:
    def __init__(self, temperature: float, model: str, json_response: bool):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        load_config(config_path)
        self.api_key = os.environ.get("CLAUDE_API_KEY")
        self.headers = {
            'Content-Type': 'application/json', 
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01'
        }
        self.model_endpoint = "https://api.anthropic.com/v1/messages"
        self.temperature = temperature
        self.model = model
        self.json_response = json_response

    # @log_function(logger)
    def invoke(self, messages: List[Dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[1]["content"]

        content = f"system:{system}\n\n user:{user}"
        if self.json_response:
            content += ". Your output must be json formatted. Just return the specified json format, do not prepend your response with anything."

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 4000,
            "temperature": self.temperature,
        }

        try:
            request_response = requests.post(
                self.model_endpoint, 
                headers=self.headers, 
                data=json.dumps(payload)
            )
            
            # print("REQUEST RESPONSE", request_response.status_code)
            request_response_json = request_response.json()

            if 'content' not in request_response_json or not request_response_json['content']:
                raise ValueError("No content in response")

            response_content = request_response_json['content'][0]['text']
            
            if self.json_response:
                response = json.dumps(json.loads(response_content))
            else:
                response = response_content

            return response
        except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as e:
            error_message = f"Error in invoking model! {str(e)}"
            print("ERROR", error_message)
            return json.dumps({"error": error_message})

class GeminiModel:
    def __init__(self, temperature: float, model: str, json_response: bool):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        load_config(config_path)
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.headers = {
            'Content-Type': 'application/json'
        }
        self.model_endpoint = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={self.api_key}"
        self.temperature = temperature
        self.model = model
        self.json_response = json_response
    
    # @log_function(logger)
    def invoke(self, messages: List[Dict[str, str]]) -> str:
        time.sleep(5)
        system = messages[0]["content"]
        user = messages[1]["content"]

        content = f"system:{system}\n\nuser:{user}"
        if self.json_response:
            content += ". Your output must be JSON formatted. Just return the specified JSON format, do not prepend your response with anything."

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": content
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature
            },
        }

        if self.json_response:
            payload["generationConfig"]["response_mime_type"] = "application/json"

        try:
            request_response = requests.post(
                self.model_endpoint, 
                headers=self.headers, 
                data=json.dumps(payload)
            )
            
            # print("REQUEST RESPONSE", request_response.status_code)
            request_response_json = request_response.json()

            if 'candidates' not in request_response_json or not request_response_json['candidates']:
                raise ValueError("No content in response")

            response_content = request_response_json['candidates'][0]['content']['parts'][0]['text']
            
            if self.json_response:
                response = json.dumps(json.loads(response_content))
            else:
                response = response_content

            return response
        except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as e:
            error_message = f"Error in invoking model! {str(e)}"
            print("ERROR", error_message)
            return json.dumps({"error": error_message})

class GroqModel:
    def __init__(self, temperature: float, model: str, json_response: bool):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        load_config(config_path)
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.headers = {
            'Content-Type': 'application/json', 
            'Authorization': f'Bearer {self.api_key}'
        }
        self.model_endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.temperature = temperature
        self.model = model
        self.json_response = json_response

    # @log_function(logger)
    def invoke(self, messages: List[Dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[1]["content"]

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": f"system:{system}\n\n user:{user}"
                }
            ],
            "temperature": self.temperature,
        }

        time.sleep(5)

        if self.json_response:
            payload["response_format"] = {"type": "json_object"}

        try:
            request_response = requests.post(
                self.model_endpoint, 
                headers=self.headers, 
                data=json.dumps(payload)
            )
            
            # print("REQUEST RESPONSE", request_response.status_code)
            request_response_json = request_response.json()
            
            if 'choices' not in request_response_json or len(request_response_json['choices']) == 0:
                raise ValueError("No choices in response")

            response_content = request_response_json['choices'][0]['message']['content']
            
            if self.json_response:
                response = json.dumps(json.loads(response_content))
            else:
                response = response_content

            return response
        except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as e:
            error_message = f"Error in invoking model! {str(e)}"
            print("ERROR", error_message)
            return json.dumps({"error": error_message})

class OllamaModel:
    def __init__(self, temperature: float, model: str, json_response: bool):
        self.headers = {"Content-Type": "application/json"}
        self.model_endpoint = "http://localhost:11434/api/generate"
        self.temperature = temperature
        self.model = model
        self.json_response = json_response

    # @log_function(logger)
    def invoke(self, messages: List[Dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[1]["content"]

        payload = {
            "model": self.model,
            "prompt": user,
            "system": system,
            "stream": False,
            "temperature": self.temperature,
        }

        if self.json_response:
            payload["format"] = "json"
        
        try:
            request_response = requests.post(
                self.model_endpoint, 
                headers=self.headers, 
                data=json.dumps(payload)
            )
            
            # print("REQUEST RESPONSE", request_response)
            request_response_json = request_response.json()
            
            if self.json_response:
                response = json.dumps(json.loads(request_response_json['response']))
            else:
                response = str(request_response_json['response'])

            return response
        except (requests.RequestException, json.JSONDecodeError) as e:
            return json.dumps({"error": f"Error in invoking model! {str(e)}"})

class VllmModel:
    def __init__(self, temperature: float, model: str, model_endpoint: str, json_response: bool, stop: str = None):
        self.headers = {"Content-Type": "application/json"}
        self.model_endpoint = model_endpoint + 'v1/chat/completions'
        self.temperature = temperature
        self.model = model
        self.json_response = json_response
        # self.guided_json = guided_json
        self.stop = stop

    # @log_function(logger)
    def invoke(self, messages: List[Dict[str, str]], guided_json: dict = None) -> str:
        system = messages[0]["content"]
        user = messages[1]["content"]

        prefix = self.model.split('/')[0]

        if prefix == "mistralai":
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"system:{system}\n\n user:{user}"
                    }
                ],
                "temperature": self.temperature,
                "stop": None,
            }
        else:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system
                    },
                    {
                        "role": "user",
                        "content": user
                    }
                ],
                "temperature": self.temperature,
                "stop": self.stop,
            }

        if self.json_response:
            payload["response_format"] = {"type": "json_object"}
            payload["guided_json"] = guided_json
        
        try:
            request_response = requests.post(
                self.model_endpoint, 
                headers=self.headers, 
                data=json.dumps(payload)
            )
            
            # print("REQUEST RESPONSE", request_response)
            request_response_json = request_response.json()
            response_content = request_response_json['choices'][0]['message']['content']
            
            if self.json_response:
                response = json.dumps(json.loads(response_content))
            else:
                response = str(response_content)
            
            return response
        except (requests.RequestException, json.JSONDecodeError) as e:
            return json.dumps({"error": f"Error in invoking model! {str(e)}"})

class OpenAIModel:
    def __init__(self, temperature: float, model: str, json_response: bool):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        load_config(config_path)
        self.model_endpoint = 'https://api.openai.com/v1/chat/completions'
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        self.temperature = temperature
        self.model = model
        self.json_response = json_response

    # @log_function(logger)
    def invoke(self, messages: List[Dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[1]["content"]

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": user
                }
            ],
            "stream": False,
            "temperature": self.temperature,
        }
        
        if self.json_response:
            payload["response_format"] = {"type": "json_object"}
        
        try:
            response_dict = requests.post(self.model_endpoint, headers=self.headers, data=json.dumps(payload))
            # print("\n\nRESPONSE", response_dict)
            response_json = response_dict.json()

            if self.json_response:
                response = json.dumps(json.loads(response_json['choices'][0]['message']['content']))
            else:
                response = response_json['choices'][0]['message']['content']

            return response
        except (requests.RequestException, json.JSONDecodeError) as e:
            return json.dumps({"error": f"Error in invoking model! {str(e)}"})