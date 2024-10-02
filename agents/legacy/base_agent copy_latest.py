import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Union, TypeVar, Generic
from typing_extensions import TypedDict
from datetime import datetime
from termcolor import colored
from models.llms import (
    OllamaModel,
    OpenAIModel,
    GroqModel,
    GeminiModel,
    ClaudeModel,
    VllmModel,
    MistralModel
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a TypeVar for the state
StateT = TypeVar('StateT', bound=Dict[str, Any])

class BaseAgent(ABC, Generic[StateT]):
    def __init__(self, model: str = None, server: str = None, temperature: float = 0, 
                 model_endpoint: str = None, stop: str = None, location: str = "us", hyrbid: bool = False):
        self.model = model
        self.server = server
        self.temperature = temperature
        self.model_endpoint = model_endpoint
        self.stop = stop
        self.llm = self.get_llm()
        self.location = location
        self.hybrid = hyrbid

    
    def get_llm(self, json_model: bool = False):
        if self.server == 'openai':
            return OpenAIModel(model=self.model, temperature=self.temperature, json_response=json_model)
        elif self.server == 'ollama':
            return OllamaModel(model=self.model, temperature=self.temperature, json_response=json_model)
        elif self.server == 'vllm':
            return VllmModel(model=self.model, temperature=self.temperature, json_response=json_model,
                             model_endpoint=self.model_endpoint, stop=self.stop)
        elif self.server == 'groq':
            return GroqModel(model=self.model, temperature=self.temperature, json_response=json_model)
        elif self.server == 'claude':
            return ClaudeModel(temperature=self.temperature, model=self.model, json_response=json_model)
        elif self.server == 'mistral':
            return MistralModel(temperature=self.temperature, model=self.model, json_response=json_model)
        elif self.server == 'gemini':
            # raise ValueError(f"Unsupported server: {self.server}")
            return GeminiModel(temperature=self.temperature, model=self.model,  json_response=json_model)
        else:
            raise ValueError(f"Unsupported server: {self.server}")

    @abstractmethod
    def get_prompt(self, state: StateT = None) -> str:
        pass

    @abstractmethod
    def get_guided_json(self, state:StateT = None) -> Dict[str, Any]:
        pass

    def update_state(self, key: str, value: Union[str, dict], state: StateT = None) -> StateT:
        state[key] = value
        return state

    @abstractmethod
    def process_response(self, response: Any, user_input: str = None, state: StateT = None) -> Dict[str, Union[str, dict]]:
        pass

    @abstractmethod
    def get_conv_history(self, state: StateT = None) -> str:
        pass

    @abstractmethod
    def get_user_input(self) -> str:
        pass

    @abstractmethod
    def use_tool(self) -> Any:
        pass


    def invoke(self, state: StateT = None, human_in_loop: bool = False, user_input: str = None, final_answer: str = None) -> StateT:
        prompt = self.get_prompt(state)
        conversation_history = self.get_conv_history(state)

        if final_answer:
            print(colored(f"\n\n{final_answer}\n\n", "green"))

        if human_in_loop:
            user_input = self.get_user_input()

        messages = [
            {"role": "system", "content": f"{prompt}\n Today's date is {datetime.now()}"},
            {"role": "user", "content": f"\n{final_answer}\n" * 10 + f"{conversation_history}\n{user_input}"}
        ]

        if self.server == 'vllm':
            guided_json = self.get_guided_json(state)
            response = self.llm.invoke(messages, guided_json)
        else:
            response = self.llm.invoke(messages)

        updates = self.process_response(response, user_input, state)
        for key, value in updates.items():
            state = self.update_state(key, value, state)
        return state
    