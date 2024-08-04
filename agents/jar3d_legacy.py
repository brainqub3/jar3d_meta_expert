import json
import textwrap
import re
import logging
from termcolor import colored
from datetime import datetime
from typing import Any, Dict, Union, List
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from agents.base_agent import BaseAgent
from utils.read_markdown import read_markdown_file
from tools.rag_tool import rag_tool
from tools.advanced_scraper import scraper
from tools.google_serper import serper_search
from utils.logging import log_function, setup_logging
from utils.message_handling import get_ai_message_contents
from prompt_engineering.guided_json_lib import (
    guided_json_search_query, 
    guided_json_best_url, 
    guided_json_router_decision, 
    guided_json_parse_expert
)


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

# def extract_last_ai_message(conversation_history):
    
#     # Iterate through the list in reverse order to find the last AIMessage
#     for message in reversed(conversation_history):
#         if isinstance(message, dict) and message.get('content', '').startswith('Expert'):
#             return message['content']
    
#     # Return None if no AIMessage is found
#     return None


# def parse_expert_text(text):
#     pattern = r'(Expert\s+\w+(?:\s+\w+)*):\s*\n?\s*"""([\s\S]*?)"""'
#     match = re.search(pattern, text, re.DOTALL)
    
#     if match:
#         print("Match found!")
#         expert_designation = match.group(1)
#         content = match.group(2).strip()
#         print(f"\n\n Expert designation: {expert_designation}")
#         # print(f"\n\n Content: {content}")
#         return expert_designation
#     else:
#         print("No match found.")
#         return ""

def chat_counter(state: State) -> State:
    chat_limit = state.get("chat_limit")
    if chat_limit is None:
        chat_limit = 0
    chat_limit += 1
    state["chat_limit"] = chat_limit
    return chat_limit

def routing_function(state: State) -> str:
    if state["router_decision"]:
        return "no_tool_expert"
    else:
        return "tool_expert"

def set_chat_finished(state: State) -> bool:
    state["chat_finished"] = True
    final_response = state["meta_prompt"][-1].content
    final_response_formatted = re.sub(r'^```python[\s\S]*?```\s*', '', final_response, flags=re.MULTILINE)
    final_response_formatted = final_response_formatted.lstrip()
    print(colored(f"\n\n Jar3d👩‍💻: {final_response_formatted}", 'cyan'))

    return state

class Jar3d(BaseAgent[State]):
    def __init__(self, model: str = None, server: str = None, temperature: float = 0, 
                 model_endpoint: str = None, stop: str = None):
        super().__init__(model, server, temperature, model_endpoint, stop)
        self.llm = self.get_llm(json_model=False)

    def get_prompt(self, state:State = None) -> str:
        system_prompt = read_markdown_file('prompt_engineering/jar3d_requirements_prompt.md')
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
        user_input = state.get("user_input")

        system_prompt = self.get_prompt()
        user_input = f"previous conversation: {history}\n {system_prompt}\n cogor {user_input}"

        while True:
            history = self.get_conv_history(state)
            state = self.invoke(state=state, user_input=user_input)
            response = state['requirements_gathering'][-1]["content"]
            response = re.sub(r'^```python[\s\S]*?```\s*', '', response, flags=re.MULTILINE)
            response = response.lstrip()

            print("\n" + "="*80)  # Print a separator line
            print(colored("Jar3d:", 'cyan', attrs=['bold']))
            
            # Wrap the text to a specified width (e.g., 70 characters)
            wrapped_text = textwrap.fill(response, width=70)
            
            # Print each line with proper indentation
            for line in wrapped_text.split('\n'):
                print(colored("  " + line, 'green'))
            
            print("="*80 + "\n")  #
            user_input = self.get_user_input()
            
            if user_input == "/end":
                break
            
            user_input = f"cogor {user_input}"

        state = self.invoke(state=state, user_input=user_input)
        response = state['requirements_gathering'][-1]["content"]
        response = re.sub(r'^```python[\s\S]*?```\s*', '', response, flags=re.MULTILINE)
        response = response.lstrip()

        print("\n" + "="*80)  # Print a separator line
        print(colored("Jar3d:", 'cyan', attrs=['bold']))
        for line in wrapped_text.split('\n'):
                print(colored("  " + line, 'green'))
            
        print("="*80 + "\n")

        return state


class MetaExpert(BaseAgent[State]):
    def __init__(self, model: str = None, server: str = None, temperature: float = 0, 
                 model_endpoint: str = None, stop: str = None):
        super().__init__(model, server, temperature, model_endpoint, stop)
        self.llm = self.get_llm(json_model=False)

    def get_prompt(self, state:None) -> str:
        system_prompt = read_markdown_file('prompt_engineering/jar3d_meta_prompt.md')
        return system_prompt
        
    def process_response(self, response: Any, user_input: str, state: State = None) -> Dict[str, List[MessageDict]]:
        user_input = None
        updates_conversation_history = {
            "meta_prompt": [
                {"role": "user", "content": f"{user_input}"},
                {"role": "assistant", "content": str(response)}

            ]
        }
        return updates_conversation_history
    
    # @log_function(logger)
    def get_conv_history(self, state: State) -> str:

        expert_message_history = f"""<Ex> \n ## Your Expert Plan {state.get("expert_plan", [])} \n 
        ## Your Expert Research {state.get("expert_research", [])} \n ## Your Expert Writing {state.get("expert_writing", [])}
        </Ex>"""

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

        counter = chat_counter(state)  # Counts every time we invoke the Meta Agent
        recursions = 3*counter - 2
        print(colored(f"\n\n * We have envoked the Meta-Agent {counter} times.\n * we have run {recursions} max total iterations: {recursion_limit}\n\n", "green"))
        
        upper_limit_recursions = recursion_limit
        lower_limit_recursions = recursion_limit - 2

        if (recursions >= lower_limit_recursions and recursions <= upper_limit_recursions) or recursions > upper_limit_recursions :
            final_answer = "**You are being explicitly told to produce your [Type 2] work now!**"
        else:
            final_answer = None

        requirements = state['requirements_gathering'][-1].content
        formatted_requirements = '\n\n'.join(re.findall(r'```python\s*([\s\S]*?)\s*```', requirements, re.MULTILINE))

        print(colored(f"\n\n User Requirements: {formatted_requirements}\n\n", 'green'))

        state = self.invoke(state=state, user_input=formatted_requirements, final_answer=final_answer)


        # print(f"\n\nDEBUG: {state}\n\n")
        meta_prompt_cor = state['meta_prompt'][-1]["content"]

        # meta_prompt_cor_formatted = '\n\n'.join(re.findall(r'```python\s*([\s\S]*?)\s*```', meta_prompt_cor, re.MULTILINE))

        print(colored(f"\n\n Meta-Prompt Chain of Reasoning: {meta_prompt_cor}\n\n", 'green'))
        
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
        
    def process_response(self, response: Any, user_input: str = None, state: State = None) -> Dict[str, Union[str, dict]]:

        meta_prompts = state.get("meta_prompt", [])
        associated_meta_prompt = meta_prompts[-1].content
        parse_expert = self.get_llm(json_model=True)

        parse_expert_prompt = """
        You must parse the expert from the text. The expert will be one of the following.
        1. Expert Planner
        2. Expert Writer
        Return your response as the following JSON
        {{"expert": "Expert Planner" or "Expert Writer"}}
        """

        input = [
                {"role": "user", "content": associated_meta_prompt},
                {"role": "assistant", "content": f"system_prompt:{parse_expert_prompt}"}

            ]


        retries = 0
        associated_expert = None

        while retries < 4 and associated_expert is None:
            retries += 1    
            if self.server == 'vllm':
                guided_json = guided_json_parse_expert
                parse_expert_response = parse_expert.invoke(input, guided_json)
            else:
                parse_expert_response = parse_expert.invoke(input)

            associated_expert_json = json.loads(parse_expert_response)
            associated_expert = associated_expert_json.get("expert")

        # associated_expert = parse_expert_text(associated_meta_prompt)
        print(colored(f"\n\n Expert: {associated_expert}\n\n", 'green'))

        if associated_expert == "Expert Planner":
            expert_update_key = "expert_plan"
        if associated_expert == "Expert Writer":
            expert_update_key = "expert_writing"

        updates_conversation_history = {
            "conversation_history": [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": f"{str(response)}"}

            ],
            expert_update_key: {"role": "assistant", "content": f"{str(response)}"}

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
        # chat_counter(state)
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
        
    def process_response(self, response: Any, user_input: str = None, state: State = None) -> Dict[str, Union[str, dict]]:

        updates_conversation_history = {
            "conversation_history": [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": f"{str(response)}"}

            ],
            "expert_research": {"role": "assistant", "content": f"{str(response)}"}
        }
        return updates_conversation_history
    
    def get_conv_history(self, state: State) -> str:
        pass
    
    def get_user_input(self) -> str:
        pass
    
    def get_guided_json(self, state: State) -> Dict[str, Any]:
        pass

    def use_tool(self, mode: str, tool_input: str, query: str = None) -> Any:
        if mode == "serper":
            results = serper_search(tool_input)
            return results
        elif mode == "rag":
            results = rag_tool(url=tool_input, query=query)
            return results

    # @log_function(logger)
    def run(self, state: State) -> State:

        # counter = chat_counter(state)

        refine_query_template = """
        # Objective
        Your mission is to systematically address your manager's instructions by determining 
        the most appropriate search query to use in the Google search engine.
        You use a flexible search algorithm to do this.

        # Manger's Instructions
        {manager_instructions}

        # Your Previous Search Queries
        {previous_search_queries}

        # Flexible Search Algorithm for Simple and Complex Questions

            1. Initial search:
            - For a simple question: "[Question keywords]"
            - For a complex topic: "[Main topic] overview"

            2. For each subsequent search:
            - You can only see the previous search query, not the current one.
            - Choose one of these strategies based on the previous query:

            a. Specify:
                Add a more specific term or aspect related to the previous query.

            b. Broaden:
                Remove a specific term or add "general" or "overview" to the previous query.

            c. Pivot:
                Choose a different but related term from the previous query.

            d. Compare:
                Add "vs" or "compared to" along with a related term.

            e. Question:
                Rephrase the previous query as a question by adding "what", "how", "why", etc.

            3. Every 5 searches:
            - Return to the original question or main topic to reset the search path.

            4. Continue until you believe you've covered the topic sufficiently or reached a set number of searches.

        # Response Format

        **Return the following JSON:**
        {{"search_query": Algorithmically refined search query.}}

        Remember:
        - You cannot see or recall any search results.
        - You can only see the immediately preceding search query, not the current one.
        - Each new search must be based solely on the terms used in your previous search.
        - Adjust your strategy based on whether you're addressing a simple question or exploring a complex topic.
            """

        best_url_template = """
            Given the serper results, and the instructions from your manager. Select the best URL

            # Manger Instructions
            {manager_instructions}

            # Serper Results
            {serper_results}

            **Return the following JSON:**


            {{"best_url": The URL of the serper results that aligns most with the instructions from your manager.,
            "pdf": A boolean value indicating whether the URL is a PDF or not. This should be True if the URL is a PDF, and False otherwise.}}

        """
        meta_prompt = state["meta_prompt"][-1].content
        previous_search_queries = state.get("previous_search_queries", [])
        # state = self.invoke(state=state, user_input=user_input, counter=counter)
        # full_query = state["conversation_history"][-1].get("content")

        refine_query = self.get_llm(json_model=True)
        refine_prompt = refine_query_template.format(manager_instructions=meta_prompt, previous_search_queries=previous_search_queries)
        input = [
                {"role": "user", "content": "Get the search query"},
                {"role": "assistant", "content": f"system_prompt:{refine_prompt}"}
            ]
        
        if self.server == 'vllm':
            guided_json = guided_json_search_query
            refined_query = refine_query.invoke(input, guided_json)
        else:
            refined_query = refine_query.invoke(input)

        refined_query_json = json.loads(refined_query)
        refined_query = refined_query_json.get("search_query")

        print(colored(f"\n\n Refined Search Query: {refined_query}\n\n", 'green'))

        state["previous_search_queries"] = refined_query
        serper_response = self.use_tool("serper", refined_query)

        best_url = self.get_llm(json_model=True)
        best_url_prompt = best_url_template.format(manager_instructions=meta_prompt, serper_results=serper_response)
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

        meta_prompt = state["meta_prompt"][-1].content
        scraper_response = self.use_tool("rag", tool_input=best_url, query=meta_prompt)
        updates = self.process_response(scraper_response, user_input="Research")

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
        
    def process_response(self, response: Any, user_input: str = None, state: State = None) -> Dict[str, Union[str, dict]]:
        updates_conversation_history = {
            "router_decision": [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": f"{str(response)}"}

                # {"role": "assistant", "content": f"<Ex>{str(response)}</Ex> Todays date is {datetime.now()}"}

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

        # update counter
        # chat_counter(state)

        router_template = """
            Given these instructions from your manager.

            # Response from Manager
            {manager_response}

            **Return the following JSON:**

            {{""router_decision: Return the next agent to pass control to.}}

            **strictly** adhere to these **guidelines** for routing.
            If your manager's response directly references the Expert Internet Researcher, return "tool_expert".
            If your manager's response does not directly reference the Expert Internet Researcher, return "no_tool_expert".
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
            # print(f"\n\n Guided JSON: {guided_json}\n\n JSON TYPE:{type(guided_json)}\n\n")
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
        "temperature": 0.2
    }

    # For OpenAI
    # agent_kwargs = {
    #     "model": "gpt-4o",
    #     "server": "openai",
    #     "temperature": 0.2
    # }

    # Ollama
    # agent_kwargs = {
    #     "model": "phi3:instruct",
    #     "server": "ollama",
    #     "temperature": 0.5
    # }

    # Groq
    # agent_kwargs = {
    #     "model": "llama3-groq-70b-8192-tool-use-preview",
    #     "server": "groq",
    #     "temperature": 0
    # }

    # # Gemnin - Not currently working, I will be debugging this soon.
    # agent_kwargs = {
    #     "model": "gemini-1.5-pro",
    #     "server": "gemini",
    #     "temperature": 0.5
    # }

    # Vllm
    # agent_kwargs = {
    #     "model": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    #     "server": "vllm",
    #     "temperature": 0,
    #     "model_endpoint": "https://j72ip7q799tpr8-8000.proxy.runpod.net/",
    # }

    tools_router_agent_kwargs = agent_kwargs.copy()
    tools_router_agent_kwargs["temperature"] = 0

    def routing_function(state: State) -> str:
        decision = state["router_decision"]
        print(colored(f"\n\n Routing function called. Decision: {decision}\n\n", 'green'))
        return decision

    graph = StateGraph(State)

    graph.add_node("jar3d", lambda state: Jar3d(**agent_kwargs).run(state=state))
    graph.add_node("meta_expert", lambda state: MetaExpert(**agent_kwargs).run(state=state))
    graph.add_node("router", lambda state: Router(**tools_router_agent_kwargs).run(state=state))
    graph.add_node("no_tool_expert", lambda state: NoToolExpert(**agent_kwargs).run(state=state))
    graph.add_node("tool_expert", lambda state: ToolExpert(**tools_router_agent_kwargs).run(state=state))
    graph.add_node("end_chat", lambda state: set_chat_finished(state))

    graph.set_entry_point("jar3d")
    graph.set_finish_point("end_chat")

    graph.add_edge("jar3d", "meta_expert")
    graph.add_edge("meta_expert", "router")
    graph.add_edge("tool_expert", "meta_expert")
    graph.add_edge("no_tool_expert", "meta_expert")
    graph.add_conditional_edges(
        "router",
        lambda state: routing_function(state),
    )
    workflow = graph.compile()

    recursion_limit = 30
    state["recursion_limit"] = recursion_limit
    state["user_input"] = "/start"
    limit = {"recursion_limit": recursion_limit + 30} # Required as a buffer.

    for event in workflow.stream(state, limit):
        pass