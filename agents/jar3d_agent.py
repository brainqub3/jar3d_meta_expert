import json
import os
from multiprocessing import Pool, cpu_count
# import requests
# from tenacity import RetryError
import concurrent.futures  # Add this import at the top of your file
import re
import logging
import chainlit as cl
from termcolor import colored
from typing import Any, Dict, Union, List
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from agents.base_agent import BaseAgent
from utils.read_markdown import read_markdown_file
from tools.google_serper import serper_search, serper_shopping_search
from utils.logging import log_function, setup_logging
from tools.offline_graph_rag_tool import run_rag
from prompt_engineering.guided_json_lib import (
    guided_json_search_query, 
    guided_json_best_url_two, 
    guided_json_router_decision, 
    guided_json_parse_expert,
    guided_json_search_query_two
)


setup_logging(level=logging.INFO)
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
    expert_research_shopping: Annotated[List[str], add_messages]
    expert_writing: str
    user_input: Annotated[List[str], add_messages]
    previous_search_queries: Annotated[List[dict], add_messages]
    router_decision: str
    chat_limit: int
    chat_finished: bool
    recursion_limit: int
    final_answer: str
    previous_type2_work: Annotated[List[str], add_messages]
    progress_tracking: str

state: State = {
    "meta_prompt": [],
    "conversation_history": [],
    "requirements_gathering": [],
    "expert_plan": [],
    "expert_research": [],
    "expert_research_shopping": [],
    "expert_writing": [],
    "user_input": [],
    "previous_search_queries": [],
    "router_decision": None,
    "chat_limit": None,
    "chat_finished": False,
    "recursion_limit": None,
    "final_answer": None,
    "previous_type2_work": [],
    "progress_tracking": None
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


def format_final_response(final_response: str) -> str:
    print(colored(f"\n\n DEBUG FINAL RESPONSE: {final_response}\n\n", 'green'))

    # Split the response at ">> FINAL ANSWER:"
    parts = final_response.split(">> FINAL ANSWER:")
    if len(parts) > 1:
        answer_part = parts[1].strip()

        # Remove any triple quotes
        final_response_formatted = answer_part.strip('"""')

        # Remove leading and trailing whitespace
        final_response_formatted = final_response_formatted.strip()

        # Remove the CoR dictionary at the end
        cor_pattern = r'\nCoR\s*=\s*\{[\s\S]*\}\s*$'
        final_response_formatted = re.sub(cor_pattern, '', final_response_formatted)

        # Remove any trailing whitespace
        final_response_formatted = final_response_formatted.rstrip()

        return final_response_formatted
    else:
        error_message = "Error: Could not find '>> FINAL ANSWER:' in the response"
        print(colored(error_message, 'red'))
        return "Error: No final answer found"


def set_chat_finished(state: State) -> State:
    state["chat_finished"] = True
    final_response = state["meta_prompt"][-1].content

    # Use the formatting function
    final_response_formatted = format_final_response(final_response)

    agent_memory_dir = '/app/agent_memory'  # No change needed
    file_path = os.path.join(agent_memory_dir, 'jar3d_final_response_previous_run.txt')

    # Save the formatted final response to a text file
    with open(file_path, 'w') as file:
        file.write(final_response_formatted)
    
    # Print confirmation message
    print(colored(f"\n\nFinal response saved to jar3d_final_response_previous_run.txt", 'green'))

    # Print the formatted final response
    print(colored(f"\n\n Jar3d👩‍💻: {final_response_formatted}", 'cyan'))

    # Update the state with the final answer
    # state["final_answer"] = final_response_formatted

    return state


class Jar3d(BaseAgent[State]):
    def __init__(self, model: str = None, server: str = None, temperature: float = 0, 
                 model_endpoint: str = None, stop: str = None):
        super().__init__(model, server, temperature, model_endpoint, stop)
        self.llm = self.get_llm(json_model=False)

    def get_prompt(self, state: State = None) -> str:
        system_prompt_md = read_markdown_file('prompt_engineering/jar3d_requirements_prompt.md')

        final_answer = None
        agent_memory_dir = '/app/agent_memory'
        file_path = os.path.join(agent_memory_dir, 'jar3d_final_response_previous_run.txt')

        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                final_answer = file.read().strip()
        
        # Add the final_answer to the system prompt if it exists
        if final_answer:
            system_prompt = f"{system_prompt_md}\n # The AI Agent's Previous Work \n <Type2> {final_answer} </Type2>"
            print(colored(f"\n\n DEBUG JAR3D SYSTEM PROMPT FINAL ANSWER: {final_answer}\n\n", 'green'))
        else:
            system_prompt = system_prompt_md
        
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
        # system_prompt = self.get_prompt()
        # user_input = f"{system_prompt}\n cogor {user_message}"
        user_input = f"cogor: {user_message}"
    
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
        system_prompt = read_markdown_file('prompt_engineering/jar3d_meta_prompt.md')
        return system_prompt
        
    def process_response(self, response: Any, user_input: str, state: State = None) -> Dict[str, List[MessageDict]]:
        user_input = None
        # Identify the type of work and expert (if applicable) from the response

        response_str = str(response)
        formatted_response = None

        if ">> FINAL ANSWER:" in response_str:
        # It's a Type 2 work - Jar3d is delivering a final answer
            next_steps = "Jar3d has delivered a final answer"
            formatted_response = format_final_response(response_str)
        else:
            # Try to extract the expert's name for Type 1 work
            expert_match = re.search(r"Expert\s+([\w\s]+):", response_str)
            if expert_match:
                # It's a Type 1 work - Jar3d is allocating an expert
                associated_expert = expert_match.group(1).strip()
                next_steps = f"Jar3d has allocated {associated_expert} to work on your request."
            else:
                # Neither Type 1 nor Type 2 work detected
                next_steps = "Jar3d is processing the request."

        updates_conversation_history = {
            "meta_prompt": [
                {"role": "user", "content": f"{user_input}"},
                {"role": "assistant", "content": str(response)}

            ],
            "conversation_history": [
                {"role": "user", "content": f"{user_input}"},
                {"role": "assistant", "content": str(response)}

            ],

            "progress_tracking": f"{next_steps}",
            "final_answer": formatted_response
    
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

        max_length = 350000
        truncated_expert_research = all_expert_research[:max_length]

        expert_message_history = f"""
        <expert_plan>
        ## Your Expert Plan:\n{state.get("expert_plan", [])}\n
        </expert_plan>

        <expert_writing>
        ## Your Expert Writing:\n{state.get("expert_writing", [])}\n
        </expert_writing>

        <internet_research_shopping_list> 
        ## Your Expert Shopping List:\n{state.get("expert_research_shopping", [])}\n
        </internet_research_shopping_list>

        <internet_research>
        ## Your Expert Research:{truncated_expert_research}\n
        </internet_research>
        """

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
            
            # print(colored(f"\n\n DEBUG Meta-Prompt: {meta_prompt}\n\n", 'yellow'))

            cor_match = '\n\n'.join(re.findall(r'```python\s*([\s\S]*?)\s*```', meta_prompt, re.MULTILINE))

            # print(colored(f"\n\n DEBUG CoR Match: {cor_match}\n\n", 'yellow'))
            user_input = f"<requirements>{formatted_requirements}</requirements> \n\n Here is your last CoR {cor_match} update your next CoR from here."
        else:
            user_input = formatted_requirements

        state = self.invoke(state=state, user_input=user_input, final_answer=final_answer)

        meta_prompt_cor = state['meta_prompt'][-1]["content"]

        print(colored(f"\n\n Meta-Prompt Chain of Reasoning: {meta_prompt_cor}\n\n", 'green'))
        
        return state
    

class NoToolExpert(BaseAgent[State]):
    print(colored(f"\n\n DEBUG: We are running the NoToolExpert tool\n\n", 'red'))
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
            expert_update_key: {"role": "assistant", "content": f"{str(response)}"},
            "progress_tracking": f"Jar3d has completed its {associated_expert} work"
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

        print(colored(f"\n\n DEBUG: We are running the NoToolExpert tool\n\n", 'red'))
        state = self.invoke(state=state, user_input=user_input)        
        return state
    

class ToolExpert(BaseAgent[State]):
    print(colored(f"\n\n DEBUG: We are running the ToolExpert tool\n\n", 'red'))
    def __init__(self, model: str = None, server: str = None, temperature: float = 0, 
                 model_endpoint: str = None, stop: str = None, location: str = None, hybrid: bool = False):
        super().__init__(model, server, temperature, model_endpoint, stop, location, hybrid)

        # print(f"\n\n DEBUG LOCATION: {self.location}")

        self.llm = self.get_llm(json_model=False)

    def get_prompt(self, state) -> str:
        system_prompt = state["meta_prompt"][-1].content
        return system_prompt
        
    def process_response(self, response: Any, user_input: str = None, state: State = None) -> Dict[str, Union[str, dict]]:

        if self.hybrid:
            message = f"""Jar3d has completed its internet research.
            Jar3d has generated a knowledge graph, you can view it here: https://neo4j.com/product/auradb/
            """
        else:
            message = f"""Jar3d has completed its internet research.
            """
        updates_conversation_history = {
            "conversation_history": [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": f"{str(response)}"}
            ],
            "expert_research": {"role": "assistant", "content": f"{str(response)}"},
            "progress_tracking": message
        }
        return updates_conversation_history
    
    def get_conv_history(self, state: State) -> str:
        pass
    
    def get_user_input(self) -> str:
        pass
    
    def get_guided_json(self, state: State) -> Dict[str, Any]:
        pass

    def use_tool(self, mode: str, engine: str, tool_input: str, meta_prompt: str = None, query: list[str] = None, hybrid: bool = False) -> Any:
        if mode == "serper":
            if engine == "search":
                results = serper_search(tool_input, self.location)
                return {"results": results, "is_shopping": False}
            elif engine == "shopping":
                results = serper_shopping_search(tool_input, self.location)
                return {"results": results, "is_shopping": True}
            # elif engine == "scholar":
            #     results = serper_scholar_search(tool_input, self.location)
            #     return {"results": results, "is_shopping": False}
            
        elif mode == "rag":
            print(colored(f"\n\n DEBUG: We are running the Graph RAG TOOL!!\n\n", 'red'))
            nodes = None
            relationships = None
            print(colored(f"\n\n DEBUG Retreival Mode: {hybrid}\n\n", 'green'))
            results = run_rag(urls=tool_input, allowed_nodes=nodes, allowed_relationships=relationships, query=query, hybrid=self.hybrid)

            return {"results": results, "is_shopping": False}

    def generate_search_queries(self, meta_prompt: str, num_queries: int = 5) -> List[Dict[str, str]]:

        print(colored(f"\n\n DEBUG: We are running the generate_search_queries tool\n\n", 'red'))
        
        refine_query_template = """
        # Objective
        Your mission is to systematically address your manager's instructions by determining 
        the most appropriate search queries to use **AND** to determine the best engine to use for each query.
        Your engine choice is either "search" or "shopping". You must return either "search" or "shopping" for each query.
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
                {{"engine": "search", "query": "Query 1"}},
                {{"engine": "shopping", "query": "Query 2"}},
                ...
                {{"engine": "search", "query": "Query {num_queries}"}}
            ]
        }}


        Remember:
        - Generate {num_queries} unique and diverse search queries.
        - Each query should explore a different aspect or approach to the topic.
        - Ensure the queries cover various aspects of the manager's instructions.
        - The "engine" field should be "search" or "shopping" for each query.
        - "search" best for general websearch.
        - "shopping" best when you need to find products and prices.
        """

        refine_query = self.get_llm(json_model=True)
        refine_prompt = refine_query_template.format(manager_instructions=meta_prompt, num_queries=num_queries)
        input_data = [
            {"role": "user", "content": "Generate search queries"},
            {"role": "assistant", "content": f"system_prompt:{refine_prompt}"}
        ]
        
        guided_json = guided_json_search_query_two

        if self.server == 'vllm':
            refined_queries = refine_query.invoke(input_data, guided_json)
        else:
            print(colored(f"\n\n DEBUG: We are running the refine_query tool without vllm\n\n", 'red'))
            refined_queries = refine_query.invoke(input_data)

        refined_queries_json = json.loads(refined_queries)
        return refined_queries_json.get("search_queries", [])

    def process_serper_result(self, query: Dict[str, str], serper_response: Dict[str, Any]) -> Dict[str, Any]:

        print(colored(f"\n\n DEBUG: We are running the process_serper_result tool\n\n", 'red'))

        best_url_template = """
            Given the Serper results and the search query, select the best URL.

            # Search Query
            {search_query}

            # Serper Results
            {serper_results}

            **Return the following JSON:**

            {{"best_url": The URL from the Serper results that aligns most with the search query.}}
        """

        best_url = self.get_llm(json_model=True)
        best_url_prompt = best_url_template.format(search_query=query["query"], serper_results=serper_response)
        input_data = [
            {"role": "user", "content": serper_response},
            {"role": "assistant", "content": f"system_prompt:{best_url_prompt}"}
        ]
        
        guided_json = guided_json_best_url_two

        if self.server == 'vllm':
            best_url = best_url.invoke(input_data, guided_json)
        else:
            print(colored(f"\n\n DEBUG: We are running the best_url tool without vllm\n\n", 'red'))
            best_url = best_url.invoke(input_data)

        best_url_json = json.loads(best_url)

        return {"query": query, "url": best_url_json.get("best_url")}

    def analyze_and_refine_queries(
        self,
        serper_results: List[Dict[str, Any]],
        meta_prompt: str,
        num_queries: int = 1  # Default to 1 query
    ) -> List[Dict[str, str]]:
        """
        Analyzes the search results and generates refined search queries.
        """

        print(colored(f"\n\n DEBUG: We are running the analyze_and_refine_queries tool\n\n", 'red'))

        observations = []
        for result in serper_results:
            results_content = result.get("results", {})
            if result.get("is_shopping"):
                # Handle shopping results if necessary
                shopping_results = results_content.get("shopping_results", [])
                snippets = [f"{item.get('title', '')} - {item.get('price', '')}" for item in shopping_results]
            else:
                # Handle organic search results
                organic_results = results_content.get("organic_results", [])
                snippets = [item.get("snippet", "") for item in organic_results]
            observations.extend(snippets)

        # Include num_queries in the prompt to control the number of queries generated
        analysis_prompt_template = """
        Based on the following search results, generate {num_queries} new search queries to further investigate the topic.

        # Search Results
        {observations}

        # Manager's Instructions
        {meta_prompt}

        # Guidelines
        - Identify gaps or missing information in the current search results.
        - Generate queries that could fill these gaps or provide deeper insight.
        - Provide diverse and relevant queries.

        Provide the new search queries in a JSON format:
        {{
            "search_queries": [
                {{"engine": "search", "query": "New Query 1"}},
                {{"engine": "shopping", "query": "New Query 2"}},
                ...
                {{"engine": "search", "query": "New Query {num_queries}"}}
            ]
        }}
        """

        analysis_prompt = analysis_prompt_template.format(
            observations="\n".join(observations),
            meta_prompt=meta_prompt,
            num_queries=num_queries  # Pass the num_queries to the prompt
        )

        analysis_llm = self.get_llm(json_model=True)
        input_data = [
            {"role": "user", "content": "Analyze and refine search queries"},
            {"role": "assistant", "content": f"system_prompt:{analysis_prompt}"}
        ]

        guided_json = guided_json_search_query_two

        if self.server == 'vllm':
            refined_queries = analysis_llm.invoke(input_data, guided_json)
        else:
            print(colored("\n\n DEBUG: We are running the analysis without vllm\n\n", 'red'))
            refined_queries = analysis_llm.invoke(input_data)

        # Parse the LLM's response
        refined_queries_json = json.loads(refined_queries)
        refined_queries_list = refined_queries_json.get("search_queries", [])

        # Limit the number of queries returned to num_queries
        return refined_queries_list[:num_queries]

    def run(self, state: State) -> State:
        meta_prompt = state["meta_prompt"][-1].content
        print(colored(f"\n\n Meta-Prompt: {meta_prompt}\n\n", 'green'))

        # Set up iterative search parameters
        max_iterations = 5  # Define a maximum number of iterations to prevent infinite loops
        iteration = 0

        # Initial search queries
        search_queries = self.generate_search_queries(meta_prompt, num_queries=5)
        all_serper_results = []
        all_best_urls = []

        while iteration < max_iterations:
            print(colored(f"\n\n Iteration {iteration + 1}\n\n", 'yellow'))
            iteration += 1

            # Use ThreadPoolExecutor to call Serper tool for each query in parallel
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(search_queries), 5)) as executor:
                    future_to_query = {
                        executor.submit(
                            self.use_tool,
                            "serper",
                            query["engine"],
                            query["query"],
                            None
                        ): query for query in search_queries
                    }
                    serper_results = []
                    for future in concurrent.futures.as_completed(future_to_query):
                        query = future_to_query[future]
                        try:
                            result = future.result()
                            serper_results.append(result)
                        except Exception as exc:
                            print(colored(f"Error processing query {query}: {exc}", 'red'))
                            serper_results.append(None)
            except Exception as e:
                print(colored(f"Error in threading: {str(e)}. Falling back to non-parallel processing.", 'red'))
                serper_results = [self.use_tool("serper", query["engine"], query["query"], None) for query in search_queries]

            # Collect and store all results
            all_serper_results.extend(zip(search_queries, serper_results))

            # Process Serper results to get best URLs
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(serper_results), 5)) as executor:
                    future_to_query = {
                        executor.submit(
                            self.process_serper_result,
                            query,
                            result["results"] if result else {}
                        ): query for query, result in zip(search_queries, serper_results)
                    }
                    best_url_results = []
                    for future in concurrent.futures.as_completed(future_to_query):
                        query = future_to_query[future]
                        try:
                            result = future.result()
                            best_url_results.append(result)
                        except Exception as exc:
                            print(colored(f"Error processing result for query {query}: {exc}", 'red'))
                            best_url_results.append(None)
            except Exception as e:
                print(colored(f"Error in threading: {str(e)}. Falling back to non-parallel processing for best URLs.", 'red'))
                best_url_results = [
                    self.process_serper_result(query, result["results"] if result else {})
                    for query, result in zip(search_queries, serper_results)
                ]

            # Collect all best URLs
            all_best_urls.extend(best_url_results)

            # Remove duplicates while preserving query alignment
            url_query_pairs = []
            seen_urls = set()
            for item in all_best_urls:
                url = item["url"]
                query = item["query"]["query"]
                engine = item["query"]["engine"]
                if url and engine == "search" and url not in seen_urls:
                    url_query_pairs.append({"url": url, "query": query})
                    seen_urls.add(url)

            # Extract unique URLs and queries while preserving alignment
            unique_urls = [item["url"] for item in url_query_pairs]
            unique_queries = [item["query"] for item in url_query_pairs]

            print(colored("\n\n Sourced data from {} sources:".format(len(unique_urls)), 'yellow'))
            print(colored(f"\n\n Search Queries {unique_queries}", 'yellow'))

            for i, url in enumerate(unique_urls, 1):
                print(colored("  {}. {}".format(i, url), 'green'))

            # Analyze search results and refine the queries
            refined_search_queries = self.analyze_and_refine_queries(
                [result for _, result in all_serper_results],
                meta_prompt,
                num_queries=1  # Limit to 1 query per iteration
            )

            # Check if refinement is needed
            if not refined_search_queries or refined_search_queries == search_queries:
                # No further refinement possible
                break

            # Update search queries for the next iteration
            search_queries = refined_search_queries

        # After iterations, process the collected results
        try:
            scraper_response = self.use_tool(
                mode="rag",
                engine=None,
                tool_input=unique_urls,
                meta_prompt=meta_prompt,
                query=unique_queries  # Pass aligned queries
            )
        except Exception as e:
            scraper_response = {"results": f"Error {e}: Failed to scrape results", "is_shopping": False}

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

            ],
            "progress_tracking": f"Jar3d has routed to an expert 🤓"
        },

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
            If your maneger's response contains "Expert Internet Researcher", return "tool_expert".
            If your manager's response contains "Expert Planner" or "Expert Writer", return "no_tool_expert".
            If your manager's response contains '>> FINAL ANSWER:', return "end_chat".

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
