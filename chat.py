import re
import asyncio
import chainlit as cl
from typing import Dict, Any
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from termcolor import colored
from typing import Union
from agents.jar3d_cl import (State, 
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


def extract_final_answer(final_answer: str) -> str:
    # Split the string at ">> FINAL ANSWER:"
    parts = final_answer.split(">> FINAL ANSWER:")
    
    if len(parts) < 2:
        raise ValueError("Could not find '>> FINAL ANSWER:' in the string")
    
    # Take the part after ">> FINAL ANSWER:"
    answer_part = parts[1].strip()
    
    # Remove the triple quotes at the start and end
    answer_part = answer_part.strip('"""')
    
    return answer_part.strip()


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

        return jar3d_intro


@cl.on_chat_start
async def start():
    # state = State()  # Initialize your State object
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
    "final_answer": None,
    }

    # For OpenAI
    # agent_kwargs = {
    #     "model": "gpt-4o",
    #     "server": "openai",
    #     "temperature": 0.1
    # }

    # Claude 
    agent_kwargs = {
        "model": "claude-3-5-sonnet-20240620",
        "server": "claude",
        "temperature": 0.1
    }

        # Ollama
    # agent_kwargs = {
    #     "model": "phi3:instruct",
    #     "server": "ollama",
    #     "temperature": 0.1
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
    #     "temperature": 0.1
    # }

    # Vllm
    # agent_kwargs = {
    #     "model": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    #     "server": "vllm",
    #     "temperature": 0.1,
    #     "model_endpoint": "https://b1xkdmrlxy9q8s-8000.proxy.runpod.net/",
    # }


    jar3d_intro = Jar3dIntro(**agent_kwargs)
    jar3d_intro_hi = jar3d_intro.run(state)
    jar3d_agent = Jar3d(**agent_kwargs)
    cl.user_session.set("agent_kwargs", agent_kwargs)
    cl.user_session.set("state", state)
    cl.user_session.set("jar3d_agent", jar3d_agent)
    
    # Send an initial message to start the conversation
    await cl.Message(content=jar3d_intro_hi, author="Jar3d👩‍💻").send()


def run_workflow(state):
    agent_kwargs = cl.user_session.get("agent_kwargs")
    tools_router_agent_kwargs = agent_kwargs.copy()
    tools_router_agent_kwargs["temperature"] = 0

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

    checkpointer = MemorySaver()
    workflow = graph.compile(checkpointer)

    recursion_limit = 4
    state["recursion_limit"] = recursion_limit
    state["user_input"] = "/start"
    configs = {"recursion_limit": recursion_limit + 10, "configurable": {"thread_id": 42}}

    final_answer = None
    for event in workflow.stream(state, configs):
        pass

    state = workflow.get_state(configs)
    state = state.values
    final_answer = extract_final_answer(state["final_answer"])
    return final_answer


@cl.on_message
async def main(message: cl.Message):
    state: State = cl.user_session.get("state")
    agent: Jar3d = cl.user_session.get("jar3d_agent")
    
    # Running the synchronous function in a separate thread
    loop = asyncio.get_running_loop()
    state, response = await loop.run_in_executor(None, agent.run_chainlit, state, message)

    # Display the response (requirements) immediately
    await cl.Message(content=response, author="Jar3d👩‍💻").send()

    if message.content == "/end":
        await cl.Message(content="Workflow compiling...", author="System").send()
        final_answer = await cl.make_async(run_workflow)(state)
        if final_answer:

            await cl.Message(content=final_answer, author="Jar3d👩‍💻").send()
        else:
            await cl.Message(content="No final answer was produced.", author="Jar3d👩‍💻").send()

    
    else:
        cl.user_session.set("state", state)  # Update the state in the session

    
if __name__ == "__main__":
    cl.run()
