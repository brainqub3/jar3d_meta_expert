# Script for registering agents.

from typing import TypedDict, Annotated, List, Any
from langgraph.graph.message import add_messages

# add_message creates this  [HumanMessage(content='Hello', id='1'), AIMessage(content='Hi there!', id='2')]
from typing import TypedDict, Annotated, Any
from langgraph.graph.message import add_messages



# Define AgentWorkpad as a TypedDict with total=False to allow extra keys
class AgentRegistry(TypedDict, total=False):
    # WebSearchAgent: Annotated[Any, add_messages]
    user: List[Any]
    # Jar3d: Annotated[Any, add_messages]
    # RAGAgent: Annotated[Any, add_messages]
    # total=False allows us to add additional agents dynamically

# Initialize the agent_workpad as an empty AgentWorkpad
# agent_workpad: AgentWorkpad = {}
AgentRegistry = {}
# AgentWorkpad is a shared dictionary instance










