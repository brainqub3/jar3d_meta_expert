import json
import os
import re
import asyncio
from termcolor import colored
from langgraph.graph import StateGraph, END, START, MessagesState
from agents.agent_base import BaseAgent
import chainlit as cl
from langgraph.checkpoint.memory import MemorySaver
from agents.agent_base import MetaAgent, SimpleAgent
from agents.agent_workpad import create_state_typed_dict
from agents.agent_registry import AgentRegistry
from agents.serper_dev_agent import SerperDevAgent
from agents.web_scraper_agent import WebScraperAgent
from agents.agent_base import ReporterAgent
from workflow_builders.meta_agent import build_workflow

@cl.on_chat_start
async def start():
    # Create the TaskList
    task_list = cl.TaskList()
    task_list.status = "Ready"
    await task_list.send()
    cl.user_session.set("task_list", task_list)

    cl.user_session.set("conversation_history", [])

    meta_agent = MetaAgent(
        name="meta_agent",
        server="claude",
        model="claude-3-5-sonnet-20240620",
        temperature=0
    )
    serper_agent = SerperDevAgent(
        name="serper_agent",
        server="claude",
        model="claude-3-5-sonnet-20240620",
        temperature=0
    )
    web_scraper_agent = WebScraperAgent(
        name="web_scraper_agent",
        server="claude",
        model="claude-3-5-sonnet-20240620",
        temperature=0
    )
    reporter_agent = ReporterAgent(
        name="reporter_agent",
        server="claude",
        model="claude-3-5-sonnet-20240620",
        temperature=0
    )

    llm = SimpleAgent(
        name="chat_model",
        server="claude",
        model="claude-3-5-sonnet-20240620",
        temperature=0
    )

    chat_model = llm.get_llm()

    prompt_path = os.path.join(
        os.path.dirname(__file__), 'prompt_engineering', 'jar3d_requirements_prompt.MD'
    )

    with open(prompt_path, 'r', encoding='utf-8') as file:
        system_prompt = file.read()

    cl.user_session.set("system_prompt", system_prompt)
    cl.user_session.set("chat_model", chat_model)
    cl.user_session.set("meta_agent", meta_agent)
    cl.user_session.set("serper_agent", serper_agent)
    cl.user_session.set("web_scraper_agent", web_scraper_agent)
    cl.user_session.set("reporter_agent", reporter_agent)

    instructions = "/start"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instructions}
    ]

    jar3d_intro_hi = chat_model.invoke(messages)
    
    await cl.Message(content=jar3d_intro_hi, author="Jar3d👩‍💻").send()

def build_chat_workflow(agent_team, requirements):
    workflow, state = build_workflow(agent_team, requirements)
    return workflow, state

async def run_workflow(workflow, state, configs):
    for event in workflow.stream(state, configs):
        response = state.get("meta_agent", "No response from ReporterAgent")[-1].page_content
        response_json = json.loads(response)
        message = response_json.get("step_4").get("final_draft")
        agent = response_json.get("Agent")

        node_output = next(iter(event.values()))
        reporter_agent_node = node_output.get("reporter_agent", "")

        if reporter_agent_node:
            reporter_agent_json = json.loads(reporter_agent_node[-1].page_content)
            message = reporter_agent_json.get("step_4").get("final_draft")

        truncated_message = message[:50]

        task_tracking_message = f"Meta Agent asked {agent} to: {truncated_message}. Task is complete."
        print(colored(f"\n\nMeta Agent asked {agent} to: {message}\n\n", 'green'))

        # Retrieve the TaskList from the user session
        task_list = cl.user_session.get("task_list")
        
        # Create a new task and add it to the task list
        new_task = cl.Task(
            title=task_tracking_message,
            status=cl.TaskStatus.DONE
        )

        await task_list.add_task(new_task)
        # Update the task list status if needed
        task_list.status = "Done"
        await task_list.send()

    return message, state, task_tracking_message

@cl.on_message
async def main(message: cl.Message):
    # Retrieve session variables
    meta_agent = cl.user_session.get("meta_agent")
    serper_agent = cl.user_session.get("serper_agent")
    web_scraper_agent = cl.user_session.get("web_scraper_agent")
    reporter_agent = cl.user_session.get("reporter_agent")
    chat_model = cl.user_session.get("chat_model")
    system_prompt = cl.user_session.get("system_prompt")
    conversation_history = cl.user_session.get("conversation_history", [])  # Default to empty list if not set
    state = cl.user_session.get("state")

    if state:
        reporter_agent_work = state.get("reporter_agent", "No response from ReporterAgent")[-1].page_content
        reporter_agent_work_json = json.loads(reporter_agent_work)
        previous_work = reporter_agent_work_json.get("step_4").get("final_draft")
        system_prompt = f"{system_prompt}\n\nLast message from the agent:\n<prev_work>{previous_work}</prev_work>"

    agent_team = [meta_agent, serper_agent, web_scraper_agent, reporter_agent]
    configs = {"recursion_limit": 30, "configurable": {"thread_id": 42}}

    # Append the new user message to the conversation history
    conversation_history.append({"role": "user", "content": message.content})

    # Prepare messages for the chat model, including the full conversation history
    messages = [
        {"role": "system", "content": system_prompt},
    ] + conversation_history  # Include the full conversation history

    chat_model_response = chat_model.invoke(messages)
    await cl.Message(content=chat_model_response, author="Jar3d👩‍💻").send()

    # Append the assistant's response to the conversation history
    conversation_history.append({"role": "assistant", "content": chat_model_response})

    # Update the conversation history in the session
    cl.user_session.set("conversation_history", conversation_history)

    if message.content == "/end":
        formatted_requirements = '\n\n'.join(re.findall(r'```python\s*([\s\S]*?)\s*```', chat_model_response, re.MULTILINE))

        print(colored(f"\n\n User Requirements: {formatted_requirements}\n\n", 'green'))

        workflow, state = build_chat_workflow(agent_team, formatted_requirements)

        message, state, task_tracking_message = await run_workflow(workflow, state, configs)
        
        # Save state & workflow to session after successful run
        cl.user_session.set("state", state)
        cl.user_session.set("workflow", workflow)

        print(colored(f"\n\nDEBUG AFTER RUN STATE: {state}\n\n", 'red'))

        await cl.Message(content=message, author="Jar3d👩‍💻").send()