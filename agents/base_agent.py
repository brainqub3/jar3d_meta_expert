import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Union, TypeVar, Generic
from typing_extensions import TypedDict

from models.llms import (
    OllamaModel,
    OpenAIModel,
    GroqModel,
    GeminiModel,
    ClaudeModel,
    VllmModel
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a TypeVar for the state
StateT = TypeVar('StateT', bound=Dict[str, Any])

class BaseAgent(ABC, Generic[StateT]):
    def __init__(self, model: str = None, server: str = None, temperature: float = 0, 
                 model_endpoint: str = None, stop: str = None):
        self.model = model
        self.server = server
        self.temperature = temperature
        self.model_endpoint = model_endpoint
        self.stop = stop
        self.llm = self.get_llm()

    
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
        elif self.server == 'gemini':
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
    def process_response(self, response: Any, user_input: str = None) -> Dict[str, Union[str, dict]]:
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


    def invoke(self, state: StateT, human_in_loop: bool = False, user_input: str = None) -> StateT:
        prompt = self.get_prompt(state)
        conversation_history = self.get_conv_history(state)

        if human_in_loop:
            user_input = self.get_user_input()

        messages = [
            {"role": "system", "content": f"{prompt}\n memory:{conversation_history}"},
            {"role": "user", "content": f"<problem>{user_input}</problem>"}
        ]

        if self.server == 'vllm':
            guided_json = self.get_guided_json(state)
            response = self.llm.invoke(messages, guided_json)
        else:
            response = self.llm.invoke(messages)

        updates = self.process_response(response, user_input)
        for key, value in updates.items():
            state = self.update_state(key, value, state)
        return state
    
