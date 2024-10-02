import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, TypeVar
from datetime import datetime
from termcolor import colored
from langchain_core.documents.base import Document
from .agent_workpad import AgentWorkpad
from .agent_registry import AgentRegistry
from models.llms import (
    OllamaModel,
    OpenAIModel,
    GroqModel,
    GeminiModel,
    ClaudeModel,
    VllmModel,
    MistralModel
)

StateT = TypeVar('StateT', bound=Dict[str, Any])

class BaseAgent(ABC, Generic[StateT]):
    """
    Abstract base class for all agents in the system.
    Provides common functionality and interface for agent implementations.
    """
    def __init__(self, name: str, model: str = None, server: str = None, temperature: float = 0, 
                 model_endpoint: str = None, stop: str = None):
        """
        Initialize the BaseAgent with common parameters.
        
        :param name: The name to register the agent
        :param model: The name of the language model to use
        :param server: The server hosting the language model
        :param temperature: Controls randomness in model outputs
        :param model_endpoint: Specific endpoint for the model API
        :param stop: Stop sequence for model generation
        """
        self.name = name  # Store the initialized name
        self.model = model
        self.server = server
        self.temperature = temperature
        self.model_endpoint = model_endpoint
        self.stop = stop
        self.llm = self.get_llm()
        self.register()

    def get_llm(self, json_response: bool = False, prompt_caching: bool = True):
        """
        Factory method to create and return the appropriate language model instance.
        :param json_response: Whether the model should return JSON responses
        :param prompt_caching: Whether to use prompt caching
        :return: An instance of the appropriate language model
        """
        if self.server == "openai":
            return OpenAIModel(
                temperature=self.temperature,
                model=self.model,
                json_response=json_response
            )
        elif self.server == "claude":
            return ClaudeModel(
                temperature=self.temperature,
                model=self.model,
                json_response=json_response,
                prompt_caching=prompt_caching
            )
        elif self.server == "mistral":
            return MistralModel(
                temperature=self.temperature,
                model=self.model,
                json_response=json_response
            )
        elif self.server == "ollama":
            return OllamaModel(
                temperature=self.temperature,
                model=self.model,
                json_response=json_response
            )
        elif self.server == "groq":
            return GroqModel(
                temperature=self.temperature,
                model=self.model,
                json_response=json_response
            )
        elif self.server == "gemini":
            return GeminiModel(
                temperature=self.temperature,
                model=self.model,
                json_response=json_response
            )
        elif self.server == "vllm":
            return VllmModel(
                temperature=self.temperature,
                model=self.model,
                model_endpoint=self.model_endpoint,
                json_response=json_response,
                stop=self.stop
            )
        else:
            raise ValueError(f"Unsupported server type: {self.server}")

    def register(self):
        """
        Register the agent in the global AgentWorkpad and AgentRegistry using its initialized name.
        Stores the agent's docstring in the AgentRegistry.
        """
        AgentWorkpad[self.name] = []  # Initialize with None or any default value
        
        # Extract the docstring from the child class
        agent_docstring = self.__class__.__doc__
        if agent_docstring:
            agent_description = agent_docstring.strip()
        else:
            agent_description = "No description provided."

        # Store the agent's description in the AgentRegistry
        if self.name != "MetaAgent":
            AgentRegistry[self.name] = agent_description
            print(f"Agent '{self.name}' registered to AgentWorkpad and AgentRegistry.")

    def write_to_workpad(self, response: Any):
        """
        Write the agent's response to the AgentWorkpad under its registered name.
        
        :param response: The response to be written to the AgentWorkpad
        """

        response_document = Document(page_content=response, metadata={"agent": self.name})
        # AgentWorkpad[self.name] = response_document

        # Ensure AgentWorkpad[self.name] is always a list
        if self.name not in AgentWorkpad or not isinstance(AgentWorkpad[self.name], list):
            AgentWorkpad[self.name] = []

        AgentWorkpad[self.name].append(response_document)
        print(f"Agent '{self.name}' wrote to AgentWorkpad.")

    def read_instructions(self, state: StateT = AgentWorkpad) -> str:
        """
        Read instructions from the MetaAgent in AgentWorkpad if the agent is not MetaAgent.
        This method can be overridden by subclasses.
        
        :param state: The current state of the agent (default is AgentWorkpad)
        :return: Instructions as a string
        """
        # if self.name != "MetaAgent":
            # Read instructions from MetaAgent's entry in the AgentWorkpad
        try:
            instructions = state.get("meta_agent", "")[-1].page_content
            print(colored(f"\n\n{self.name} read instructions from MetaAgent: {instructions}\n\n", 'green'))
        except Exception as e:
            print(f"You must have a meta_agent in your workflow: {e}")
            return ""
        return instructions
        # return ""

    @abstractmethod
    def invoke(self, state: StateT = AgentWorkpad) -> StateT:
        """
        Abstract method to invoke the agent's main functionality.
        
        :param state: The current state of the agent (default is AgentWorkpad)
        :return: Updated state after invocation
        """
        pass

class ToolCallingAgent(BaseAgent[StateT]):
    """
    An agent capable of calling external tools based on instructions.
    """
    @abstractmethod
    def get_guided_json(self, state: StateT = AgentWorkpad) -> Dict[str, Any]:
        """
        Abstract method to get guided JSON for tool calling.
        
        :param state: The current state of the agent (default is AgentWorkpad)
        :return: A dictionary representing the guided JSON
        """
        pass

    def call_tool(self, instructions: str, guided_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an external tool based on instructions and guided JSON.
        
        :param instructions: Instructions for the tool
        :param guided_json: Guided JSON for structuring the tool call
        :return: The response from the LLM as a JSON string
        """
        messages = [
            {"role": "system", "content": f"Take the following instructions and return the specified JSON: {guided_json}."},
            {"role": "user", "content": instructions}
        ]
        json_llm = self.get_llm(json_response=True)
        response = json_llm.invoke(messages)
        return response

    @abstractmethod
    def execute_tool(self, tool_response: Dict[str, Any], state: StateT = AgentWorkpad) -> StateT:
        """
        Abstract method to execute a tool based on its response.
        
        :param tool_response: The response from the called tool
        :param state: The current state of the agent (default is AgentWorkpad)
        :return: Updated state after tool execution
        """
        pass

    # def invoke(self, state: StateT = AgentWorkpad) -> StateT:
    #     """
    #     Invoke the tool calling process.
        
    #     :param state: The current state of the agent (default is AgentWorkpad)
    #     :return: Updated state after tool invocation
    #     """
    #     instructions = self.read_instructions(state)
    #     guided_json = self.get_guided_json(state)
    #     tool_response_str = self.call_tool(instructions, guided_json)
    #     # Parse the JSON string returned by LLM into a dictionary
    #     try:
    #         tool_response = json.loads(tool_response_str)
    #     except json.JSONDecodeError as e:
    #         print(f"Error parsing JSON response from LLM: {e}")
    #         raise ValueError("Invalid JSON response from LLM.") from e
    #     updated_state = self.execute_tool(tool_response, state)
    #     # Update the AgentWorkpad using the agent's initialized name
    #     self.write_to_workpad(updated_state)
    #     return updated_state

    def invoke(self, state: StateT = AgentWorkpad) -> StateT:
        """
        Invoke the agent's main functionality.
        """
        instructions = self.read_instructions(state)
        if not instructions:
            print(f"No instructions provided to {self.name}.")
            return state
        guided_json = self.get_guided_json(state)
        tool_response_str = self.call_tool(instructions, guided_json)
        # Parse the JSON string returned by LLM into a dictionary
        try:
            tool_response = json.loads(tool_response_str)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response from LLM: {e}")
            raise ValueError("Invalid JSON response from LLM.") from e
        # Execute the tool and get the results
        result = self.execute_tool(tool_response, state)
        # Update the state with the results under the agent's name
        state[self.name] = result
        print(f"DEBUG: SerperDevAgent result: {result} \n\n Type:{type(result)}")
        # Write the results to the AgentWorkpad
        self.write_to_workpad(result)
        print(f"{self.name} wrote results to AgentWorkpad.")
        return state

import os

class MetaAgent(BaseAgent[StateT]):
    """
    An agent that generates responses based on instructions and state.
    """

    def read_instructions(self, state: StateT = AgentWorkpad) -> str:
        """
        Read instructions from the 'meta_prompt.MD' file in the 'prompt_engineering' folder.

        :param state: The current state of the agent (default is AgentWorkpad)
        :return: Instructions as a string
        """
        # Construct the path to the meta_prompt.MD file
        prompt_path = os.path.join(
            os.path.dirname(__file__), '..', 'prompt_engineering', 'meta_prompt.MD'
        )
        try:
            with open(prompt_path, 'r', encoding='utf-8') as file:
                instructions = file.read()
            return instructions
        except FileNotFoundError:
            print(f"File not found: {prompt_path}")
            return ""
        except Exception as e:
            print(f"Error reading instructions from {prompt_path}: {e}")
            return ""

    def get_guided_json(self, state: StateT = AgentWorkpad) -> Dict[str, Any]:
        """
        Get guided JSON schema for response generation, aligning with meta_prompt.MD.

        :param state: The current state of the agent (default is AgentWorkpad)
        :return: Guided JSON schema as a dictionary
        """
        guided_json_schema = {
            "type": "object",
            "properties": {
                "Agent": {
                    "type": "string",
                    "description": "The agent from the registry you wish to direct"
                },
                "step_1": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "First set of actions, e.g., ['draft_response', 'self_critique']"
                },
                "step_2": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Second set of actions, e.g., ['draft_response_2', 'self_critique']"
                },
                "step_3": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Third set of actions, e.g., ['draft_response_3', 'self_critique']"
                },
                "step_4": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Final steps, e.g., ['improvements to be applied', 'final_draft']"
                }
            },
            "required": ["Agent", "step_1", "step_2", "step_3", "step_4"]
        }
        return guided_json_schema

    def respond(self, instructions: str, requirements: str, state: StateT = AgentWorkpad, agent_register: StateT = AgentRegistry) -> str:
        """
        Generate a response based on instructions and state.

        :param instructions: Instructions for generating the response
        :param state: The current state of the agent (default is AgentWorkpad)
        :return: Generated response as a string
        """
        # Unpack all key-value pairs in the state and include them in the message
        if state:
            workpad = "\n".join(f"{key}: {value}" for key, value in state.items())
        else:
            workpad = "No previous state."

        if agent_register:
            agent_register_content = "\n".join(f"{key}: {value}" for key, value in agent_register.items())
        else:
            agent_register_content = "No previous agent register."

        user_message = f"<workpad>{workpad}</workpad>\n<agent_register>{agent_register_content}</agent_register>\n<user_requirements>{requirements}</user_requirements>"

        messages = [
            {"role": "system", "content": f"{instructions} respond in the following JSON format: {self.get_guided_json(state)}"},
            {"role": "user", "content": user_message}
        ]
        json_llm = self.get_llm(json_response=True)
        response = json_llm.invoke(messages)
        return response

    def write_to_workpad(self, response: Any):
        """
        Write the response to the AgentWorkpad.

        :param response: The response to be written to the AgentWorkpad
        """
        response_json = json.loads(response)
        meta_agent_response = response_json.get("step_4", "No response generated.")[1]
        meta_agent_response_document = Document(page_content=meta_agent_response, metadata={"agent": self.name})
        # AgentWorkpad[self.name] = meta_agent_response_document

        if AgentWorkpad[self.name] is None:
            AgentWorkpad[self.name] = meta_agent_response_document
        else:
            AgentWorkpad[self.name].append(meta_agent_response_document)
        print(f"Agent '{self.name}' wrote to AgentWorkpad.")

    def invoke(self, requirements: str, state: StateT = AgentWorkpad) -> StateT:
        """
        Invoke the response generation process.

        :param state: The current state of the agent (default is AgentWorkpad)
        :return: Updated state after response generation
        """
        instructions = self.read_instructions(state)
        response = self.respond(instructions, requirements, state)
        # Update the AgentWorkpad using the agent's initialized name
        self.write_to_workpad(response)
        return state


# from typing import Dict, Any
# from .agent_base import BaseAgent, StateT
# import json

class ReporterAgent(BaseAgent[StateT]):
    """
    This agent is used to Provide your final response to the user.

    Inputs:
        - 'instruction': Your final response to the user.

    Outputs:
        - 'response': Your final response delivered to the user.
    """

    def __init__(self, name: str, model: str = "gpt-3.5-turbo", server: str = "openai", temperature: float = 0):
        super().__init__(name, model=model, server=server, temperature=temperature)
        print(f"ReporterAgent '{self.name}' initialized.")

    def read_instructions(self, state: StateT = AgentWorkpad) -> str:
        """
        Read the instruction from the AgentWorkpad.
        """
        try:
            instruction = state.get("meta_agent", "")[-1].page_content
            print(f"{self.name} read instruction: {instruction}")
        except Exception as e:
            print(f"You must have a meta_agent in your workflow: {e}")
            return ""
        return instruction

    # def respond(self, instruction: str, state: StateT = AgentWorkpad) -> str:
    #     """
    #     Generate a response based on the instruction.

    #     :param instruction: The instruction to respond to.
    #     :param state: The current state of the agent (default is AgentWorkpad)
    #     :return: Generated response as a string
    #     """
    #     messages = [
    #         {"role": "system", "content": "Please follow the instructions carefully to produce the desired output."},
    #         {"role": "user", "content": instruction}
    #     ]
    #     response = self.llm.invoke(messages)
    #     return response

    def invoke(self, state: StateT = AgentWorkpad) -> StateT:
        """
        Invoke the agent's main functionality: process the instruction and return a response.
        """
        instruction = self.read_instructions(state)
        if not instruction:
            print(f"No instruction provided to {self.name}.")
            return state

        # response = self.respond(instruction, state)
        print(f"{self.name} is reporting the response to user")

        # # Update the state with the response under the agent's name
        # state[self.name] = instruction

        # Write the response to the AgentWorkpad
        self.write_to_workpad(instruction)
        print(f"{self.name} wrote response to AgentWorkpad.")

        return state