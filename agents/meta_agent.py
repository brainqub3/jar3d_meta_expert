import json
import logging
from termcolor import colored
from datetime import datetime
from typing import Any, Dict, Union, List
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from agents.base_agent import BaseAgent
from utils.read_markdown import read_markdown_file
from tools.advanced_scraper import scraper
from tools.google_serper import serper_search
from utils.logging import log_function, setup_logging
from utils.message_handling import get_ai_message_contents
from prompt_engineering.guided_json_lib import guided_json_search_query, guided_json_best_url, guided_json_router_decision

setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MessageDict(TypedDict):
    role: str
    content: str

class State(TypedDict):
    meta_prompt: Annotated[List[MessageDict], add_messages]
    conversation_history: Annotated[List[dict], add_messages]
    user_input: Annotated[List[str], add_messages]
    router_decision: bool
    chat_limit: int
    chat_finished: bool

state: State = {
    "meta_prompt": [],
    "conversation_history": [],
    "user_input": [],
    "router_decision": None,
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

def routing_function(state: State) -> str:
    if state["router_decision"]:
        return "no_tool_expert"
    else:
        return "tool_expert"

def set_chat_finished(state: State) -> bool:
    state["chat_finished"] = True
    final_response = state["meta_prompt"][-1].content
    print(colored(f"\n\n Meta Agent 🧙‍♂️: {final_response}", 'cyan'))

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
        user_input = None
        updates_conversation_history = {
            "meta_prompt": [
                {"role": "user", "content": f"{user_input}"},
                {"role": "assistant", "content": str(response)}

            ]
        }
        return updates_conversation_history
    
    @log_function(logger)
    def get_conv_history(self, state: State) -> str:
        conversation_history = state.get("conversation_history", [])
        expert_message_history = get_ai_message_contents(conversation_history)
        print(f"Expert Data Collected: {expert_message_history}")
        expert_message_history = f"Expert Data Collected: <Ex>{expert_message_history}</Ex>"
        return expert_message_history
    
    def get_user_input(self) -> str:
        user_input = input("Enter your query: ")
        return user_input
    
    def get_guided_json(self, state: State) -> Dict[str, Any]:
        pass

    def use_tool(self) -> Any:
        pass

    @log_function(logger)
    def run(self, state: State) -> State:

        user_input = state.get("user_input")
        state = self.invoke(state=state, user_input=user_input)
        
        return state
    

class NoToolExpert(BaseAgent[State]):
    def __init__(self, model: str = None, server: str = None, temperature: float = 0, 
                 model_endpoint: str = None, stop: str = None):
        super().__init__(model, server, temperature, model_endpoint, stop)
        self.llm = self.get_llm(json_model=False)

    def get_prompt(self, state) -> str:
        # print(f"\nn{state}\n")
        system_prompt = state["meta_prompt"][-1].content
        return system_prompt
        
    def process_response(self, response: Any, user_input: str = None) -> Dict[str, Union[str, dict]]:
        updates_conversation_history = {
            "conversation_history": [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": f"{str(response)}"}

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


    # @log_function(logger)
    def run(self, state: State) -> State:
        user_input = state["meta_prompt"][1].content
        state = self.invoke(state=state, user_input=user_input)        
        return state
    

class ToolExpert(BaseAgent[State]):
    def __init__(self, model: str = None, server: str = None, temperature: float = 0, 
                 model_endpoint: str = None, stop: str = None):
        super().__init__(model, server, temperature, model_endpoint, stop)
        self.llm = self.get_llm(json_model=False)

    def get_prompt(self, state) -> str:
        system_prompt = state["meta_prompt"][-1].content
        return system_prompt
        
    def process_response(self, response: Any, user_input: str = None) -> Dict[str, Union[str, dict]]:
        updates_conversation_history = {
            "conversation_history": [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": f"{str(response)}"}

            ]
        }
        return updates_conversation_history
    
    def get_conv_history(self, state: State) -> str:
        pass
    
    def get_user_input(self) -> str:
        pass
    
    def get_guided_json(self, state: State) -> Dict[str, Any]:
        pass

    def use_tool(self, mode: str, tool_input: str, doc_type: str = None) -> Any:
        if mode == "serper":
            results = serper_search(tool_input)
            return results
        elif mode == "scraper":
            results = scraper(tool_input, doc_type)
            return results

    # @log_function(logger)
    def run(self, state: State) -> State:

        refine_query_template = """
            Given the response from your manager.

            # Response from Manager
            {manager_response}

            **Return the following JSON:**


            {{"search_query": The refined google search engine query that aligns with the response from your managers.}}

        """

        best_url_template = """
            Given the serper results, and the instructions from your manager. Select the best URL

            # Manger Instructions
            {manager_response}

            # Serper Results
            {serper_results}

            **Return the following JSON:**


            {{"best_url": The URL of the serper results that aligns most with the instructions from your manager.,
            "pdf": A boolean value indicating whether the URL is a PDF or not. This should be True if the URL is a PDF, and False otherwise.}}

        """

        user_input = state["meta_prompt"][-1].content
        state = self.invoke(state=state, user_input=user_input)
        full_query = state["conversation_history"][-1].get("content")

        refine_query = self.get_llm(json_model=True)
        refine_prompt = refine_query_template.format(manager_response=full_query)
        input = [
                {"role": "user", "content": full_query},
                {"role": "assistant", "content": f"system_prompt:{refine_prompt}"}

            ]
        
        if self.server == 'vllm':
            guided_json = guided_json_search_query
            refined_query = refine_query.invoke(input, guided_json)
        else:
            refined_query = refine_query.invoke(input)

        refined_query_json = json.loads(refined_query)
        refined_query = refined_query_json.get("search_query")
        serper_response = self.use_tool("serper", refined_query)

        best_url = self.get_llm(json_model=True)
        best_url_prompt = best_url_template.format(manager_response=full_query, serper_results=serper_response)
        input = [
                {"role": "user", "content": serper_response},
                {"role": "assistant", "content": f"system_prompt:{best_url_prompt}"}

            ]
        
        if self.server == 'vllm':
            guided_json = guided_json_best_url
            best_url = best_url.invoke(input, guided_json)
        else:
            best_url = best_url.invoke(input)

        best_url_json = json.loads(best_url)
        best_url = best_url_json.get("best_url")

        doc_type = best_url_json.get("pdf")

        if doc_type == "True" or doc_type == True:
            doc_type = "pdf"
        else:
            doc_type = "html"

        scraper_response = self.use_tool("scraper", best_url, doc_type)
        updates = self.process_response(scraper_response, user_input)

        for key, value in updates.items():
            state = self.update_state(key, value, state)
                
        return state
    
class Router(BaseAgent[State]):
    def __init__(self, model: str = None, server: str = None, temperature: float = 0, 
                 model_endpoint: str = None, stop: str = None):
        super().__init__(model, server, temperature, model_endpoint, stop)
        self.llm = self.get_llm(json_model=True)

    def get_prompt(self, state) -> str:
        system_prompt = state["meta_prompt"][-1].content
        return system_prompt
        
    def process_response(self, response: Any, user_input: str = None) -> Dict[str, Union[str, dict]]:
        updates_conversation_history = {
            "router_decision": [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": f"<Ex>{str(response)}</Ex> Todays date is {datetime.now()}"}

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
        pass

    # @log_function(logger)
    def run(self, state: State) -> State:

        router_template = """
            Given these instructions from your manager.

            # Response from Manager
            {manager_response}

            **Return the following JSON:**

            {{""router_decision: Return the next agent to pass control to.}}

            **strictly** adhere to these **guidelines** for routing.
            If your manager's response suggests a tool might be required to answer the query, return "tool_expert".
            If your manager's response suggests no tool is required to answer the query, return "no_tool_expert".
            If your manager's response suggest they have provided a final answer, return "end_chat".

        """
        system_prompt = router_template.format(manager_response=state["meta_prompt"][-1].content)
        input = [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": f"system_prompt:{system_prompt}"}

            ]
        router = self.get_llm(json_model=True)

        if self.server == 'vllm':
            guided_json = guided_json_router_decision
            router_response = router.invoke(input, guided_json)
        else:
            router_response = router.invoke(input)

        router_response = json.loads(router_response)
        router_response = router_response.get("router_decision")
        state = self.update_state("router_decision", router_response, state)
        
        return state
    
# Example usage
if __name__ == "__main__":
    from langgraph.graph import StateGraph


    # For Claude
    agent_kwargs = {
        "model": "claude-3-5-sonnet-20240620",
        "server": "claude",
        "temperature": 0.5
    }

    # For OpenAI
    # agent_kwargs = {
    #     "model": "gpt-4o",
    #     "server": "openai",
    #     "temperature": 0.5
    # }

    # Ollama
    # agent_kwargs = {
    #     "model": "phi3:instruct",
    #     "server": "ollama",
    #     "temperature": 0.5
    # }

    # Groq
    # agent_kwargs = {
    #     "model": "mixtral-8x7b-32768",
    #     "server": "groq",
    #     "temperature": 0.5
    # }

    # # Gemnin - Not currently working, I will be debugging this soon.
    # agent_kwargs = {
    #     "model": "gemini-1.5-pro",
    #     "server": "gemini",
    #     "temperature": 0.5
    # }

    # # Vllm
    # agent_kwargs = {
    #     "model": "meta-llama/Meta-Llama-3-70B-Instruct",
    #     "server": "vllm",
    #     "temperature": 0.5,
    #     "model_endpoint": "https://vpzatdgopr2pmx-8000.proxy.runpod.net/",
    # }

    tools_router_agent_kwargs = agent_kwargs.copy()
    tools_router_agent_kwargs["temperature"] = 0

    def routing_function(state: State) -> str:
        decision = state["router_decision"]
        print(colored(f"\n\n Routing function called. Decision: {decision}", 'red'))
        return decision

    graph = StateGraph(State)

    graph.add_node("meta_expert", lambda state: MetaExpert(**agent_kwargs).run(state=state))
    graph.add_node("router", lambda state: Router(**tools_router_agent_kwargs).run(state=state))
    graph.add_node("no_tool_expert", lambda state: NoToolExpert(**agent_kwargs).run(state=state))
    graph.add_node("tool_expert", lambda state: ToolExpert(**tools_router_agent_kwargs).run(state=state))
    graph.add_node("end_chat", lambda state: set_chat_finished(state))

    graph.set_entry_point("meta_expert")
    graph.set_finish_point("end_chat")

    graph.add_edge("meta_expert", "router")
    graph.add_edge("tool_expert", "meta_expert")
    graph.add_edge("no_tool_expert", "meta_expert")
    graph.add_conditional_edges(
        "router",
        lambda state: routing_function(state),
    )
    workflow = graph.compile()

    while True:
        query = input("Ask me anything: ")
        if query.lower() == "exit":
            break

        # current_time = datetime.now()
        state["user_input"] = query
        limit = {"recursion_limit": 30}

        for event in workflow.stream(state, limit):
            pass