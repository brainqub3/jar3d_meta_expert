import re
import logging
from agents.base_agent import BaseAgent
from typing import List, Dict, Any
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from utils.read_markdown import read_markdown_file
from utils.logging import log_function, setup_logging
from termcolor import colored
import textwrap

 

setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MessageDict(TypedDict):
    role: str
    content: str

class State(TypedDict):
    meta_prompt: Annotated[List[MessageDict], add_messages]
    conversation_history: Annotated[List[dict], add_messages]
    requirements_gathering: Annotated[List[str], add_messages]
    expert_plan: str
    expert_research: Annotated[List[str], add_messages]
    expert_writing: str
    user_input: Annotated[List[str], add_messages]
    previous_search_queries: Annotated[List[dict], add_messages]
    router_decision: bool
    chat_limit: int
    chat_finished: bool
    recursion_limit: int

state: State = {
    "meta_prompt": [],
    "conversation_history": [],
    "requirements_gathering": [],
    "expert_plan": [],
    "expert_research": [],
    "expert_writing": [],
    "user_input": [],
    "previous_search_queries": [],
    "router_decision": None,
    "chat_limit": None,
    "chat_finished": False,
    "recursion_limit": None
}


class MetaExpert(BaseAgent[State]):
    def __init__(self, model: str = None, server: str = None, temperature: float = 0, 
                 model_endpoint: str = None, stop: str = None):
        super().__init__(model, server, temperature, model_endpoint, stop)
        self.llm = self.get_llm(json_model=False)

    def get_prompt(self, state:State = None) -> str:
        system_prompt = read_markdown_file('prompt_engineering/requirements_gathering_prompt.md')
        return system_prompt
        
    def process_response(self, response: Any, user_input: str, state: State = None) -> Dict[str, List[MessageDict]]:
        user_input = None
        updates_conversation_history = {
            "requirements_gathering": [
                {"role": "user", "content": f"{user_input}"},
                {"role": "assistant", "content": str(response)}

            ]
        }
        return updates_conversation_history
    
    # @log_function(logger)
    def get_conv_history(self, state: State) -> str:

        conversation_history = state.get('requirements_gathering', [])

        return conversation_history
    
    def get_user_input(self) -> str:
        user_input = input("Enter your query: ")
        return user_input
    
    def get_guided_json(self, state: State) -> Dict[str, Any]:
        pass

    def use_tool(self) -> Any:
        pass

    @log_function(logger)
    def run(self, state: State) -> State:
        history = self.get_conv_history(state)
        user_input = "/start"
        system_prompt = self.get_prompt()

        user_input = f"previous conversation: {history}\n {system_prompt}\n {user_input}"

        while user_input != "/end":
            history = self.get_conv_history(state)
            state = self.invoke(state=state, user_input=user_input)
            # print(colored(f"\n\n{state['requirements_gathering'][-1]} \n\n", 'green'))
            response = state['requirements_gathering'][-1]['content']
            response = re.sub(r'^```python[\s\S]*?```\s*', '', response, flags=re.MULTILINE)
            response = response.lstrip()

            print("\n" + "="*80)  # Print a separator line
            print(colored("Assistant:", 'cyan', attrs=['bold']))
            
            # Wrap the text to a specified width (e.g., 70 characters)
            wrapped_text = textwrap.fill(response, width=70)
            
            # Print each line with proper indentation
            for line in wrapped_text.split('\n'):
                print(colored("  " + line, 'green'))
            
            print("="*80 + "\n")  #
            user_input = self.get_user_input()


        state = self.invoke(state=state, user_input=user_input)

        print(f"\n\nSTATE: {state}\n\n")
        
        return state
    
agent_kwargs = {
        "model": "gpt-4o",
        "server": "openai",
        "temperature": 0
    }


agent = MetaExpert(**agent_kwargs)
state = agent.run(state)
