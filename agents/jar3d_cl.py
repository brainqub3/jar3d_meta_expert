import json
from multiprocessing import Pool, cpu_count
import requests
from tenacity import RetryError
import re
import logging
import chainlit as cl
from termcolor import colored
from typing import Any, Dict, Union, List
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from agents.base_agent import BaseAgent
from utils.read_markdown import read_markdown_file
from tools.google_serper import serper_search
from utils.logging import log_function, setup_logging
from tools.offline_rag_tool import run_rag
from prompt_engineering.guided_json_lib import (
    guided_json_search_query, 
    guided_json_best_url_two, 
    guided_json_router_decision, 
    guided_json_parse_expert
)


setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MessageDict(TypedDict):
    role: str
    content: str

class State(TypedDict):
    meta_prompt: Annotated[List[dict], add_messages]
    conversation_history: Annotated[List[dict], add_messages]
    requirements_gathering: Annotated[List[str], add_messages]
    expert_plan: str
    expert_research: Annotated[List[str], add_messages]
    expert_writing: str
    user_input: Annotated[List[str], add_messages]
    previous_search_queries: Annotated[List[dict], add_messages]
    router_decision: str
    chat_limit: int
    chat_finished: bool
    recursion_limit: int
    final_answer: str

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
    "recursion_limit": None,
    "final_answer": None
}

def chat_counter(state: State) -> State:
    chat_limit = state.get("chat_limit")
    if chat_limit is None:
        chat_limit = 0
    chat_limit += 1
    state["chat_limit"] = chat_limit
    return chat_limit

def routing_function(state: State) -> str:
        decision = state["router_decision"]
        print(colored(f"\n\n Routing function called. Decision: {decision}\n\n", 'green'))
        return decision

def set_chat_finished(state: State) -> bool:
    state["chat_finished"] = True
    final_response = state["meta_prompt"][-1].content
    final_response_formatted = re.sub(r'^```python[\s\S]*?```\s*', '', final_response, flags=re.MULTILINE)
    final_response_formatted = final_response_formatted.lstrip()
    print(colored(f"\n\n Jar3d👩‍💻: {final_response_formatted}", 'cyan'))
    state["final_answer"] = final_response_formatted

    return state

class Jar3d(BaseAgent[State]):
    def __init__(self, model: str = None, server: str = None, temperature: float = 0, 
                 model_endpoint: str = None, stop: str = None):
        super().__init__(model, server, temperature, model_endpoint, stop)
        self.llm = self.get_llm(json_model=False)

    def get_prompt(self, state: State = None) -> str:

        if self.server == 'claude':
            system_prompt = read_markdown_file('prompt_engineering/jar3d_requirements_prompt.xml')
        else:   
            system_prompt = read_markdown_file('prompt_engineering/jar3d_requirements_prompt.md')
        return system_prompt
        
    def process_response(self, response: Any, user_input: str, state: State = None) -> Dict[str, List[Dict[str, str]]]:
        updates_conversation_history = {
            "requirements_gathering": [
                {"role": "user", "content": f"{user_input}"},
                {"role": "assistant", "content": str(response)}
            ]
        }
        return updates_conversation_history
    
    def get_conv_history(self, state: State) -> str:
        conversation_history = state.get('requirements_gathering', [])
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
    
    def get_user_input(self) -> str:
        pass

    def get_guided_json(self, state: State) -> Dict[str, Any]:
        pass

    def use_tool(self) -> Any:
        pass

    def run_chainlit(self, state: State, message: cl.Message) -> State:
        user_message = message.content
        system_prompt = self.get_prompt()
        user_input = f"{system_prompt}\n cogor {user_message}"
    
        state = self.invoke(state=state, user_input=user_input)
        response = state['requirements_gathering'][-1]["content"]
        response = re.sub(r'^```python[\s\S]*?```\s*', '', response, flags=re.MULTILINE)
        response = response.lstrip()

        return state, response


class MetaExpert(BaseAgent[State]):
    def __init__(self, model: str = None, server: str = None, temperature: float = 0, 
                 model_endpoint: str = None, stop: str = None):
        super().__init__(model, server, temperature, model_endpoint, stop)
        self.llm = self.get_llm(json_model=False)

    def get_prompt(self, state:None) -> str:
        if self.server == 'claude':
            system_prompt = read_markdown_file('prompt_engineering/jar3d_meta_prompt.xml')
        else:
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

        all_expert_research = []

        if state["expert_research"]:
            expert_research = state["expert_research"]
            all_expert_research.extend(expert_research)
        else:
            all_expert_research = []

        expert_message_history = f"""<Ex> \n ## Your Expert Plan {state.get("expert_plan", [])} \n 
        ## Your Expert Research {all_expert_research} \n ## Your Expert Writing {state.get("expert_writing", [])}
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
        recursion_limit = state.get("recursion_limit")
        recursions = 3*counter - 2
        print(colored(f"\n\n * We have envoked the Meta-Agent {counter} times.\n * we have run {recursions} max total iterations: {recursion_limit}\n\n", "green"))
        
        upper_limit_recursions = recursion_limit
        lower_limit_recursions = recursion_limit - 2

        if recursions >= lower_limit_recursions and recursions <= upper_limit_recursions:
            final_answer = "**You are being explicitly told to produce your [Type 2] work now!**"
        elif recursions > upper_limit_recursions:
            extra_recursions = recursions - upper_limit_recursions
            base_message = "**You are being explicitly told to produce your [Type 2] work now!**"
            final_answer = (base_message + "\n") * (extra_recursions + 1)
        else:
            final_answer = None

        try:
            requirements = state['requirements_gathering'][-1]["content"]
        except: 
            requirements = state['requirements_gathering'][-1].content

        formatted_requirements = '\n\n'.join(re.findall(r'```python\s*([\s\S]*?)\s*```', requirements, re.MULTILINE))

        print(colored(f"\n\n User Requirements: {formatted_requirements}\n\n", 'green'))

        if state.get("meta_prompt"):
            try:
                meta_prompt = state['meta_prompt'][-1]["content"]
            except:
                meta_prompt = state['meta_prompt'][-1].content
            
            cor_match = re.search(r'(CoR\s*=\s*\{[^}]+\})', meta_prompt, re.DOTALL)
            cor_string = cor_match.group(1)
            user_input = f"{formatted_requirements}\n\n Here is your last CoR {cor_string} update your CoR from here."
        else:
            user_input = formatted_requirements

        state = self.invoke(state=state, user_input=user_input, final_answer=final_answer)

        meta_prompt_cor = state['meta_prompt'][-1]["content"]

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

        # meta_prompts = state.get("meta_prompt", [])
        associated_meta_prompt = state["meta_prompt"][-1].content
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
        all_expert_research = []
        meta_prompt = state["meta_prompt"][1].content

        if state.get("expert_research"):
            expert_research = state["expert_research"]
            all_expert_research.extend(expert_research)
            research_prompt = f"\n Your response must be delivered considering following research.\n ## Research\n {all_expert_research} "
            user_input = f"{meta_prompt}\n{research_prompt}"

        else:
            user_input = meta_prompt

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
            results = run_rag(urls=tool_input, query=query)
            return results

    def generate_search_queries(self, meta_prompt: str, num_queries: int = 5) -> List[str]:
        refine_query_template = """
        # Objective
        Your mission is to systematically address your manager's instructions by determining 
        the most appropriate search queries to use in the Google search engine.
        You will generate {num_queries} different search queries.

        # Manager's Instructions
        {manager_instructions}

        # Flexible Search Algorithm for Simple and Complex Questions

            1. Initial search:
            - For a simple question: "[Question keywords]"
            - For a complex topic: "[Main topic] overview"

            2. For each subsequent search:
            - Choose one of these strategies:

            a. Specify:
                Add a more specific term or aspect related to the topic.

            b. Broaden:
                Remove a specific term or add "general" or "overview" to the query.

            c. Pivot:
                Choose a different but related term from the topic.

            d. Compare:
                Add "vs" or "compared to" along with a related term.

            e. Question:
                Rephrase the query as a question by adding "what", "how", "why", etc.

        # Response Format

        **Return the following JSON:**
        {{
            "search_queries": [
                "Query 1",
                "Query 2",
                ...
                "Query {num_queries}"
            ]
        }}

        Remember:
        - Generate {num_queries} unique and diverse search queries.
        - Each query should explore a different aspect or approach to the topic.
        - Ensure the queries cover various aspects of the manager's instructions.
        """

        refine_query = self.get_llm(json_model=True)
        refine_prompt = refine_query_template.format(manager_instructions=meta_prompt, num_queries=num_queries)
        input = [
            {"role": "user", "content": "Generate search queries"},
            {"role": "assistant", "content": f"system_prompt:{refine_prompt}"}
        ]
        
        guided_json = guided_json_search_query

        if self.server == 'vllm':
            refined_queries = refine_query.invoke(input, guided_json)
        else:
            refined_queries = refine_query.invoke(input)

        refined_queries_json = json.loads(refined_queries)
        return refined_queries_json.get("search_queries", [])

    def process_serper_result(self, args):
        query, serper_response = args
        best_url_template = """
            Given the serper results, and the search query, select the best URL

            # Search Query
            {search_query}

            # Serper Results
            {serper_results}

            **Return the following JSON:**

            {{"best_url": The URL of the serper results that aligns most with the search query.}}
        """

        best_url = self.get_llm(json_model=True)
        best_url_prompt = best_url_template.format(search_query=query, serper_results=serper_response)
        input = [
            {"role": "user", "content": serper_response},
            {"role": "assistant", "content": f"system_prompt:{best_url_prompt}"}
        ]
        
        guided_json = guided_json_best_url_two

        if self.server == 'vllm':
            best_url = best_url.invoke(input, guided_json)
        else:
            best_url = best_url.invoke(input)

        best_url_json = json.loads(best_url)
        return best_url_json.get("best_url")

    def run(self, state: State) -> State:
        meta_prompt = state["meta_prompt"][-1].content
        print(colored(f"\n\n Meta-Prompt: {meta_prompt}\n\n", 'green'))

        # Generate multiple search queries
        search_queries = self.generate_search_queries(meta_prompt, num_queries=20)
        print(colored(f"\n\n Generated Search Queries: {search_queries}\n\n", 'green'))

        try:
            # Use multiprocessing to call Serper tool for each query in parallel
            with Pool(processes=min(cpu_count(), len(search_queries))) as pool:
                serper_results = pool.starmap(self.use_tool, [("serper", query) for query in search_queries])

            # Process Serper results to get best URLs
            with Pool(processes=min(cpu_count(), len(serper_results))) as pool:
                best_urls = pool.map(self.process_serper_result, zip(search_queries, serper_results))
        except Exception as e:
            print(colored(f"Error in multithreaded processing: {str(e)}. Falling back to non-multithreaded approach.", "yellow"))
            # Fallback to non-multithreaded approach
            serper_results = [self.use_tool("serper", query) for query in search_queries]
            best_urls = [self.process_serper_result((query, result)) for query, result in zip(search_queries, serper_results)]

        # Remove duplicates from the list of URLs
        unique_urls = list(dict.fromkeys(url for url in best_urls if url))

        print(colored("\n\n Sourced data from {} sources:".format(len(unique_urls)), 'green'))
        for i, url in enumerate(unique_urls, 1):
            print(colored("  {}. {}".format(i, url), 'green'))
        print()

        scraper_response = self.use_tool("rag", tool_input=unique_urls, query=meta_prompt)
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
            router_response = router.invoke(input, guided_json)
        else:
            router_response = router.invoke(input)

        router_response = json.loads(router_response)
        router_response = router_response.get("router_decision")

        state = self.update_state("router_decision", router_response, state)
   
        return state