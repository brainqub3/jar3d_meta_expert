import requests
import time
import json
import os
import logging
from typing import List, Dict
from utils.logging import log_function, setup_logging
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from config.load_configs import load_config

setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BaseModel:
    def __init__(self, temperature: float, model: str, json_response: bool, max_retries: int = 3, retry_delay: int = 1):
        self.temperature = temperature
        self.model = model
        self.json_response = json_response
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type(requests.RequestException))
    def _make_request(self, url, headers, payload):
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    
class MistralModel(BaseModel):
    def __init__(self, temperature: float, model: str, json_response: bool, max_retries: int = 3, retry_delay: int = 1):
        super().__init__(temperature, model, json_response, max_retries, retry_delay)
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        load_config(config_path)
        self.api_key = os.environ.get("MISTRAL_API_KEY")
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        self.model_endpoint = "https://api.mistral.ai/v1/chat/completions"

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type(requests.RequestException))
    def _make_request(self, url, headers, payload):
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()

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
            "temperature": self.temperature,
        }

        if self.json_response:
            payload["response_format"] = {"type": "json_object"}

        try:
            request_response_json = self._make_request(self.model_endpoint, self.headers, payload)
            
            if 'choices' not in request_response_json or len(request_response_json['choices']) == 0:
                raise ValueError("No choices in response")

            response_content = request_response_json['choices'][0]['message']['content']
            
            if self.json_response:
                response = json.dumps(json.loads(response_content))
            else:
                response = response_content

            return response
        except requests.RequestException as e:
            return json.dumps({"error": f"Error in invoking model after {self.max_retries} retries: {str(e)}"})
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            return json.dumps({"error": f"Error processing response: {str(e)}"})


class ClaudeModel(BaseModel):
    def __init__(self, temperature: float, model: str, json_response: bool, max_retries: int = 3, retry_delay: int = 1):
        super().__init__(temperature, model, json_response, max_retries, retry_delay)
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        load_config(config_path)
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.headers = {
            'Content-Type': 'application/json', 
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01'
        }
        self.model_endpoint = "https://api.anthropic.com/v1/messages"

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        # time.sleep(5)
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
            "max_tokens": 4096,
            "temperature": self.temperature,
        }

        try:
            request_response_json = self._make_request(self.model_endpoint, self.headers, payload)
            
            if 'content' not in request_response_json or not request_response_json['content']:
                raise ValueError("No content in response")

            response_content = request_response_json['content'][0]['text']
            
            if self.json_response:
                response = json.dumps(json.loads(response_content))
            else:
                response = response_content

            return response
        except requests.RequestException as e:
            return json.dumps({"error": f"Error in invoking model after {self.max_retries} retries: {str(e)}"})
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            return json.dumps({"error": f"Error processing response: {str(e)}"})

class GeminiModel(BaseModel):
    def __init__(self, temperature: float, model: str, json_response: bool, max_retries: int = 3, retry_delay: int = 1):
        super().__init__(temperature, model, json_response, max_retries, retry_delay)
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        load_config(config_path)
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.headers = {
            'Content-Type': 'application/json'
        }
        self.model_endpoint = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={self.api_key}"

    def invoke(self, messages: List[Dict[str, str]]) -> str:
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
                        "response_mime_type": "application/json",
                        "temperature": self.temperature
                    },
                }
            # payload["generationConfig"]["response_mime_type"] = "application/json"

        try:
            request_response_json = self._make_request(self.model_endpoint, self.headers, payload)

            if 'candidates' not in request_response_json or not request_response_json['candidates']:
                raise ValueError("No content in response")

            response_content = request_response_json['candidates'][0]['content']['parts'][0]['text']
            
            if self.json_response:
                response = json.dumps(json.loads(response_content))
            else:
                response = response_content

            return response
        except requests.RequestException as e:
            return json.dumps({"error": f"Error in invoking model after {self.max_retries} retries: {str(e)}"})
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            return json.dumps({"error": f"Error processing response: {str(e)}"})

class GroqModel(BaseModel):
    def __init__(self, temperature: float, model: str, json_response: bool, max_retries: int = 3, retry_delay: int = 1):
        super().__init__(temperature, model, json_response, max_retries, retry_delay)
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        load_config(config_path)
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.headers = {
            'Content-Type': 'application/json', 
            'Authorization': f'Bearer {self.api_key}'
        }
        self.model_endpoint = "https://api.groq.com/openai/v1/chat/completions"

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

        time.sleep(10)

        if self.json_response:
            payload["response_format"] = {"type": "json_object"}

        try:
            request_response_json = self._make_request(self.model_endpoint, self.headers, payload)
            
            if 'choices' not in request_response_json or len(request_response_json['choices']) == 0:
                raise ValueError("No choices in response")

            response_content = request_response_json['choices'][0]['message']['content']
            
            if self.json_response:
                response = json.dumps(json.loads(response_content))
            else:
                response = response_content

            return response
        except requests.RequestException as e:
            return json.dumps({"error": f"Error in invoking model after {self.max_retries} retries: {str(e)}"})
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            return json.dumps({"error": f"Error processing response: {str(e)}"})

class OllamaModel(BaseModel):
    def __init__(self, temperature: float, model: str, json_response: bool, max_retries: int = 3, retry_delay: int = 1):
        super().__init__(temperature, model, json_response, max_retries, retry_delay)
        self.headers = {"Content-Type": "application/json"}
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model_endpoint = f"{self.ollama_host}/api/generate"

    def _check_and_pull_model(self):
        # Check if the model exists
        response = requests.get(f"{self.ollama_host}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            if not any(model["name"] == self.model for model in models):
                print(f"Model {self.model} not found. Pulling the model...")
                self._pull_model()
            else:
                print(f"Model {self.model} is already available.")
        else:
            print(f"Failed to check models. Status code: {response.status_code}")

    def _pull_model(self):
        pull_endpoint = f"{self.ollama_host}/api/pull"
        payload = {"name": self.model}
        response = requests.post(pull_endpoint, json=payload, stream=True)
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    status = json.loads(line.decode('utf-8'))
                    print(f"Pulling model: {status.get('status')}")
            print(f"Model {self.model} pulled successfully.")
        else:
            print(f"Failed to pull model. Status code: {response.status_code}")

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        self._check_and_pull_model()  # Check and pull the model if necessary

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
            request_response_json = self._make_request(self.model_endpoint, self.headers, payload)
            
            if self.json_response:
                response = json.dumps(json.loads(request_response_json['response']))
            else:
                response = str(request_response_json['response'])

            return response
        except requests.RequestException as e:
            return json.dumps({"error": f"Error in invoking model after {self.max_retries} retries: {str(e)}"})
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Error processing response: {str(e)}"})
class VllmModel(BaseModel):
    def __init__(self, temperature: float, model: str, model_endpoint: str, json_response: bool, stop: str = None, max_retries: int = 5, retry_delay: int = 1):
        super().__init__(temperature, model, json_response, max_retries, retry_delay)
        self.headers = {"Content-Type": "application/json"}
        self.model_endpoint = model_endpoint + 'v1/chat/completions'
        self.stop = stop

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
            request_response_json = self._make_request(self.model_endpoint, self.headers, payload)
            response_content = request_response_json['choices'][0]['message']['content']
            
            if self.json_response:
                response = json.dumps(json.loads(response_content))
            else:
                response = str(response_content)
            
            return response
        except requests.RequestException as e:
            return json.dumps({"error": f"Error in invoking model after {self.max_retries} retries: {str(e)}"})
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Error processing response: {str(e)}"})

class OpenAIModel(BaseModel):
    def __init__(self, temperature: float, model: str, json_response: bool, max_retries: int = 3, retry_delay: int = 1):
        super().__init__(temperature, model, json_response, max_retries, retry_delay)
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        load_config(config_path)
        self.model_endpoint = 'https://api.openai.com/v1/chat/completions'
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[1]["content"]

        if self.model == "o1-preview" or self.model == "o1-mini":

            payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": f"{system}\n\n{user}"
                }
            ],
            "stream": False,
            "temperature": self.temperature,
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
            "stream": False,
            "temperature": self.temperature,
        }
        
        if self.json_response:
            payload["response_format"] = {"type": "json_object"}
        
        try:
            response_json = self._make_request(self.model_endpoint, self.headers, payload)

            if self.json_response:
                response = json.dumps(json.loads(response_json['choices'][0]['message']['content']))
            else:
                response = response_json['choices'][0]['message']['content']

            return response
        except requests.RequestException as e:
            return json.dumps({"error": f"Error in invoking model after {self.max_retries} retries: {str(e)}"})
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Error processing response: {str(e)}"})
