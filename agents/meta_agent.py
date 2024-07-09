import json
from typing import Any, Dict, Union, List
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from agents.base_agent import BaseAgent
from utils.read_markdown import read_markdown_file

class MessageDict(TypedDict):
    role: str
    content: str

class State(TypedDict):
    meta_prompt: Annotated[List[MessageDict], add_messages]
    conversation_history: Annotated[List[dict], add_messages]
    user_input: Annotated[List[str], add_messages]
    chat_limit: int
    chat_finished: bool

state: State = {
    "meta_prompt": [],
    "conversation_history": [],
    "user_input": [],
    "chat_limit": None,
    "chat_finished": False
}

def chat_counter(state: State) -> State:
    chat_limit = state.get("chat_limit")
    if chat_limit is None:
        chat_limit = 0
    chat_limit += 1
    state["chat_limit"] = chat_limit
    return state

def router(state: State, limit) -> str:
    chat_limit = state.get("chat_limit")
    reursion_limit = limit.get("recursion_limit")
    if chat_limit == reursion_limit:
        return "end_chat"
    else:
        return "professor_synapse"

def set_chat_finished(state: State) -> bool:
    state["chat_finished"] = True
    return state

class MetaExpert(BaseAgent[State]):
    def __init__(self, model: str = None, server: str = None, temperature: float = 0, 
                 model_endpoint: str = None, stop: str = None):
        super().__init__(model, server, temperature, model_endpoint, stop)
        self.llm = self.get_llm(json_model=False)

    def get_prompt(self, state:None) -> str:
        system_prompt = read_markdown_file('prompt_engineering/meta_prompt.md')
        return system_prompt
        
    def process_response(self, response: Any, user_input: str) -> Dict[str, List[MessageDict]]:
        updates_conversation_history = {
            "meta_prompt": [
                {"role": "user", "content": f"{user_input}"},
                {"role": "assistant", "content": str(response)}

            ]
        }
        return updates_conversation_history
    
    def get_conv_history(self, state: State) -> str:
        return state.get("conversation_history", [])
    
    def get_user_input(self) -> str:
        user_input = input("Enter your query: ")
        return user_input
    
    def get_guided_json(self, state: State) -> Dict[str, Any]:
        pass

    def run(self, input_dict, state: State) -> State:

        user_input = input_dict.get("user_input")
        state = self.update_state("user_input", user_input, state)

        print(f"STATE BEFORE: {state}")
        state = self.invoke(state=state, user_input=user_input)
        print(f"STATE AFTER: {state}")
        
        return state
    

class NoToolExpert(BaseAgent[State]):
    def __init__(self, model: str = None, server: str = None, temperature: float = 0, 
                 model_endpoint: str = None, stop: str = None):
        super().__init__(model, server, temperature, model_endpoint, stop)
        self.llm = self.get_llm(json_model=False)

    def get_prompt(self, state) -> str:
        print(f"\nn{state}\n")
        system_prompt = state["meta_prompt"][-1].content
        return system_prompt
        
    def process_response(self, response: Any, user_input: str = None) -> Dict[str, Union[str, dict]]:
        updates_conversation_history = {
            "conversation_history": [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": f"<Ex>{str(response)}</Ex>"}

            ]
        }
        return updates_conversation_history
    
    def get_conv_history(self, state: State) -> str:
        pass
    
    def get_user_input(self) -> str:
        pass
    
    def get_guided_json(self, state: State) -> Dict[str, Any]:
        pass

    def run(self, state: State) -> State:

        user_input = state["meta_prompt"][1].content
        print(f"\n\nUSER INPUT: {user_input}")
        print(f"STATE BEFORE: {state}")
        state = self.invoke(state=state, user_input=user_input)
        print(f"STATE AFTER: {state}")
        
        return state
    
# Example usage
if __name__ == "__main__":
    from langgraph.graph import StateGraph

    graph = StateGraph(State)
    agent_kwargs = {
        "model": "gpt-4o",
        "server": "openai",
        "temperature": 0,
    }

    query = "Write me an expertly crafted Haiku"
    input_dict = {"user_input": query}

    graph.add_node("MetaExpert", lambda state: MetaExpert(**agent_kwargs).run(state=state, input_dict=input_dict))
    graph.add_node("NoToolExpert", lambda state: NoToolExpert(**agent_kwargs).run(state=state))
    # graph.add_node("chat_counter", lambda state: chat_counter(state))
    # graph.add_node("end_chat", lambda state: set_chat_finished(state))

    graph.set_entry_point("MetaExpert")
    graph.set_finish_point("NoToolExpert")

    graph.add_edge("MetaExpert", "NoToolExpert")
    # graph.add_conditional_edges(
    #     "chat_counter",
    #     lambda state: router(state, limit),
    # )

    workflow = graph.compile()
    limit = {"recursion_limit": 10}


    for event in workflow.stream(state, limit):
        pass