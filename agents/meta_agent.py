import json
from typing import Any, Dict, Union, List
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from agents.base_agent import BaseAgent
from utils.read_markdown import read_markdown_file
from tools.basic_scraper import scrape_website
from tools.google_serper import serper_search

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

    def use_tool(self) -> Any:
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

    def use_tool(self) -> Any:
        pass

    def run(self, state: State) -> State:

        user_input = state["meta_prompt"][1].content
        
        print(f"\n\nUSER INPUT: {user_input}")
        print(f"STATE BEFORE: {state}")
        state = self.invoke(state=state, user_input=user_input)
        print(f"STATE AFTER: {state}")
        
        return state
    

class ToolExpert(BaseAgent[State]):
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

    def use_tool(self, tool_input: str, mode: str) -> Any:
        if mode == "serper":
            results = serper_search(tool_input)
            return results
        elif mode == "scraper":
            results = scrape_website(tool_input)
            return results

    def run(self, state: State) -> State:


        refine_query_template = """
            Given the response from your manager.

            # Response from Manager
            {manager_response}

            ```json
            {{"search_query": The refined google search engine query that aligns with the response from your managers.}}

        """

        best_url_template = """
            Given the serper results, and the instructions from your manager. Select the best URL
            Return the following JSON:

            # Manger Instructions
            {manager_response}

            # Serper Results
            {serper_results}

            ```json
            {{"best_url": The URL of the serper results that aligns most with the instructions from your manager.}}

        """

        user_input = state["meta_prompt"][-1].content
        state = self.invoke(state=state, user_input=user_input)
        full_query = state["conversation_history"][-1].content

        refine_query = self.get_llm(json_model=True)
        refine_prompt = refine_query_template.format(manager_response=full_query)
        input = [
                {"role": "user", "content": full_query},
                {"role": "assistant", "content": f"system_prompt:{refine_prompt}"}

            ]
        refined_query = refine_query.invoke(input)
        refined_query = refined_query["search_query"]
        serper_response = self.use_tool(refined_query, "serper")

        best_url = self.get_llm(json_model=True)
        best_url_prompt = best_url_template.format(manager_response=full_query, serper_results=serper_response)
        input = [
                {"role": "user", "content": serper_response},
                {"role": "assistant", "content": f"system_prompt:{best_url_prompt}"}

            ]
        best_url = best_url.invoke(input)
        best_url = best_url["best_url"]
        scraper_response = self.use_tool(best_url, "scraper")

        state = self.update_state("conversation_history", scraper_response, state)
        
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

    query = "Tell me a funny joke"
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