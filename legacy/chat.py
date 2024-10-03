import os
import asyncio
import re
import chainlit as cl
from typing import Dict, Any
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import Union
from chainlit.input_widget import Select
from agents.jar3d_agent import (State, 
                          Jar3d, 
                          MetaExpert, 
                          Router, 
                          NoToolExpert, 
                          ToolExpert, 
                          set_chat_finished, 
                          routing_function,
                          )
from agents.base_agent import BaseAgent
from utils.read_markdown import read_markdown_file
from config.load_configs import load_config


config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
load_config(config_path)

server = os.environ.get("LLM_SERVER")
recursion_limit = int(os.environ.get("RECURSION_LIMIT"))

def get_agent_kwargs(server: str = "claude", location: str = None, rag_mode: str = "Non-RAG") -> Dict[str, Any]:

    if not location:
        location = "us"
    else:
        location = location

    if server == "openai":
        agent_kwargs = {
        "model": "gpt-4o",
        "server": "openai",
        "temperature": 0,
        }
        agent_kwargs_meta_expert = agent_kwargs.copy()
        agent_kwargs_meta_expert["model"] = "o1-preview"

    # Mistral 
    elif server == "mistral":
        agent_kwargs = {
            "model": "mistral-large-latest",
            "server": "mistral",
            "temperature": 0,
        }
        agent_kwargs_meta_expert = agent_kwargs.copy()
    
    elif server == "claude":
        agent_kwargs = {
            "model": "claude-3-5-sonnet-20240620",
            "server": "claude",
            "temperature": 0,
        }
        agent_kwargs_meta_expert = agent_kwargs.copy()

    elif server == "ollama":
        agent_kwargs = {
            "model": os.environ.get("OLLAMA_MODEL"),
            "server": "ollama",
            "temperature": 0.1,
        }
        agent_kwargs_meta_expert = agent_kwargs.copy()

    elif server == "groq":
        agent_kwargs = {
            "model": "llama3-groq-70b-8192-tool-use-preview",
            "server": "groq",
            "temperature": 0,
        }
        agent_kwargs_meta_expert = agent_kwargs.copy()

    # you must change the model and model_endpoint to the correct values
    elif server == "vllm":
        agent_kwargs = {
            "model": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
            "server": "vllm",
            "temperature": 0.2,
            "model_endpoint": "https://s1s4l1lhce486j-8000.proxy.runpod.net/",
        }
        agent_kwargs_meta_expert = agent_kwargs.copy()

    agent_kwargs_tools = agent_kwargs.copy()
    agent_kwargs_tools["location"] = location
    agent_kwargs_tools["rag_mode"] = rag_mode

    return agent_kwargs, agent_kwargs_tools, agent_kwargs_meta_expert

class Jar3dIntro(BaseAgent[State]):
    def __init__(self, model: str = None, server: str = None, temperature: float = 0, 
                 model_endpoint: str = None, stop: str = None):
        super().__init__(model, server, temperature, model_endpoint, stop)
        self.llm = self.get_llm(json_model=False)

    def get_prompt(self, state) -> str:
        system_prompt = read_markdown_file('prompt_engineering/jar3d_requirements_prompt.md')
        return system_prompt
        
    def process_response(self, response: Any, user_input: str = None, state: State = None) -> Dict[str, Union[str, dict]]:
        user_input = "/start"
        updates_conversation_history = {
            "requirements_gathering": [
                {"role": "user", "content": f"{user_input}"},
                {"role": "assistant", "content": str(response)}

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
        state = self.invoke(state=state, user_input="/start")
        jar3d_intro = state["requirements_gathering"][-1]["content"]
        jar3d_intro = re.sub(r'^```python[\s\S]*?```\s*', '', jar3d_intro, flags=re.MULTILINE)
        jar3d_intro = jar3d_intro.lstrip() 

        return jar3d_intro

@cl.on_settings_update
async def update_settings(settings):


    location = settings["location"]
    location_dict = {
        "The United States": "us",
        "The United Kingdom": "gb",
        "The Netherlands": "nl",
        "Canada": "ca"
    }

    gl = location_dict.get(location, 'us')
    cl.user_session.set("gl", gl)

    retrieval_mode = settings["retrieval_mode"]

    if retrieval_mode == "rag_mode (Graph + Dense)":
        rag_mode = "Hybrid"
    elif retrieval_mode == "Dense Only":
        rag_mode = "Dense"
    elif retrieval_mode == "Non-RAG Search":
        rag_mode = "Non-RAG"

    cl.user_session.set("rag_mode", rag_mode)

    agent_kwargs, agent_kwargs_tools, agent_kwargs_meta_expert = get_agent_kwargs(server, gl, rag_mode)
    cl.user_session.set("agent_kwargs", agent_kwargs)
    cl.user_session.set("agent_kwargs_tools", agent_kwargs_tools)
    cl.user_session.set("agent_kwargs_meta_expert", agent_kwargs_meta_expert)

    workflow = build_workflow()
    cl.user_session.set("workflow", workflow)

    await cl.Message(content=f"I'll be conducting any Internet searches from {location} using {retrieval_mode}", author="Jar3d👩‍💻").send()



@cl.on_chat_start
async def start():

    agent_memory_dir = '/app/agent_memory'
    file_path = os.path.join(agent_memory_dir, 'jar3d_final_response_previous_run.txt')
    
    # Ensure the directory exists
    os.makedirs(agent_memory_dir, exist_ok=True)
    
    # Clear the file content
    open(file_path, 'w').close()

    task_list = cl.TaskList()
    task_list.status = "Ready"
    cl.user_session.set("task_list", task_list)

    # Send the TaskList to the UI
    await task_list.send()

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

    cl.user_session.set("state", state)

    await cl.ChatSettings(
        [
            Select(
                id="location",
                label="Select your location:",
                values=[
                    "The United States",
                    "The United Kingdom",
                    "The Netherlands",
                    "Canada",
                ],
                initial_index=2,
            ), 
            Select(
                id="retrieval_mode",
                label="Select retrieval mode:",
                values=[
                    "Hybrid (Graph + Dense)",
                    "Dense Only",
                    "Non-RAG Search"
                ],
                initial_index=2,
                description="Hybrid: Slow but more comprehensive search. Dense: Faster but less comprehensive search. Non-RAG: Fastest, no retreival potentially expensive for large context."
            )

        ]
    ).send()

    try:
        gl = cl.user_session.get("gl")
        rag_mode = cl.user_session.get("rag_mode")
    except Exception as e:
        gl = "us"
        rag_mode = False
        
    agent_kwargs, agent_kwargs_tools, agent_kwargs_meta_expert = get_agent_kwargs(server, gl, rag_mode)
    cl.user_session.set("agent_kwargs", agent_kwargs)
    cl.user_session.set("agent_kwargs_tools", agent_kwargs_tools)
    cl.user_session.set("agent_kwargs_meta_expert", agent_kwargs_meta_expert)

    workflow = build_workflow()

    cl.user_session.set("workflow", workflow)


    def initialise_jar3d():
        jar3d_intro = Jar3dIntro(**agent_kwargs)
        jar3d_intro_hi = jar3d_intro.run(state)
        jar3d_agent = Jar3d(**agent_kwargs)
        return jar3d_intro_hi, jar3d_agent
    
    loop = asyncio.get_running_loop()
    jar3d_intro_hi, jar3d_agent = await loop.run_in_executor(None, initialise_jar3d)
    cl.user_session.set("jar3d_agent", jar3d_agent)
    
    # Send an initial message to start the conversation
    await cl.Message(content=f"{jar3d_intro_hi}.\n\n I'll be conducting any Internet searches from The United States with Dense Retrieval.", author="Jar3d👩‍💻").send()


def build_workflow():
    agent_kwargs = cl.user_session.get("agent_kwargs")
    agent_kwargs_tools = cl.user_session.get("agent_kwargs_tools")
    agent_kwargs_meta_expert = cl.user_session.get("agent_kwargs_meta_expert")

    # Initialize agent instances
    meta_expert_instance = MetaExpert(**agent_kwargs_meta_expert)
    router_instance = Router(**agent_kwargs)
    no_tool_expert_instance = NoToolExpert(**agent_kwargs)
    tool_expert_instance = ToolExpert(**agent_kwargs_tools)

    graph = StateGraph(State)
    graph.add_node("meta_expert", lambda state: meta_expert_instance.run(state=state))
    graph.add_node("router", lambda state: router_instance.run(state=state))
    graph.add_node("no_tool_expert", lambda state: no_tool_expert_instance.run(state=state))
    graph.add_node("tool_expert", lambda state: tool_expert_instance.run(state=state))
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

    checkpointer = MemorySaver()
    workflow = graph.compile(checkpointer)
    return workflow

def _run_workflow_sync(workflow, state, configs, progress_queue):
    seen_progress_messages = set()
    try:
        for event in workflow.stream(state, configs):
            # Access the node's output directly
            node_output = next(iter(event.values()))

            # Access 'progress_tracking' from the node's output
            progress_message = node_output.get("progress_tracking", "")
            if progress_message:
                if progress_message not in seen_progress_messages:
                    print(f"Extracted progress_message: {progress_message}")
                    progress_queue.put_nowait(progress_message)
                    seen_progress_messages.add(progress_message)
                else:
                    print(f"Duplicate progress_message ignored: {progress_message}")
        progress_queue.put_nowait(None)  # Signal that the workflow is complete
    except Exception as e:
        print(f"Exception in workflow execution: {e}")
        progress_queue.put_nowait(None)

async def run_workflow(workflow, state):
    state["recursion_limit"] = recursion_limit
    state["user_input"] = "/start"
    configs = {"recursion_limit": recursion_limit + 10, "configurable": {"thread_id": 42}}

    progress_queue = asyncio.Queue()
    task_list = cl.user_session.get("task_list")

    # Set the TaskList status and send it to the UI
    task_list.status = "Running..."
    await task_list.send()

    loop = asyncio.get_running_loop()
    # Run the synchronous _run_workflow_sync in a background thread
    loop.run_in_executor(
        None, _run_workflow_sync, workflow, state, configs, progress_queue
    )

    # Process progress messages and update the TaskList
    while True:
        progress_message = await progress_queue.get()
        if progress_message is None:
            # Workflow is complete
            break

        # Create a new task with status RUNNING
        task = cl.Task(title=progress_message, status=cl.TaskStatus.RUNNING)
        await task_list.add_task(task)
        await task_list.send()

        # Simulate task completion (you can adjust this based on actual progress)
        task.status = cl.TaskStatus.DONE
        await task_list.send()

    # Update TaskList status to Done and send the final update
    task_list.status = "Done"
    await task_list.send()

    # Retrieve the final state
    final_state = workflow.get_state(configs)
    final_state = final_state.values
    final_answer = final_state.get(
        "final_answer",
        "The agent failed to deliver a final response. Please check the logs for more information."
    )
    return final_answer


@cl.on_message
async def main(message: cl.Message):
    state: State = cl.user_session.get("state")
    agent: Jar3d = cl.user_session.get("jar3d_agent")
    workflow = cl.user_session.get("workflow")
    
    loop = asyncio.get_running_loop()
    state, response = await loop.run_in_executor(None, agent.run_chainlit, state, message)

    await cl.Message(content=response, author="Jar3d👩‍💻").send()

    if message.content == "/end":
        await cl.Message(
            content="This will take some time, probably a good time for a coffee break ☕...",
            author="System"
        ).send()
        final_answer = await run_workflow(workflow, state)
        if final_answer:
            await cl.Message(content=final_answer, author="Jar3d👩‍💻").send()
        else:
            await cl.Message(content="No final answer was produced.", author="Jar3d👩‍💻").send()
    else:
        cl.user_session.set("state", state)

# if __name__ == "__main__":
#     cl.run()