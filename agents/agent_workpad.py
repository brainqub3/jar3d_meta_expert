# # Script for registering agents.

# from typing import TypedDict, Annotated, Any, List
# from langgraph.graph.message import add_messages

# def append_messages(left: list, right: list) -> list:
#     """
#     Appends new messages to the existing list of messages.
#     """
#     return left + right

# # Define AgentWorkpad as a TypedDict with total=False to allow extra keys
# class AgentWorkpad(TypedDict, total=False):
#     # WebSearchAgent: Annotated[Any, add_messages]
#     MetaAgent: List[Any]
#     # Jar3d: Annotated[Any, add_messages]
#     # RAGAgent: Annotated[Any, add_messages]
#     # total=False allows us to add additional agents dynamically

# # Initialize the agent_workpad as an empty AgentWorkpad
# # agent_workpad: AgentWorkpad = {}
# AgentWorkpad = {"MetaAgent":[]}
# # AgentWorkpad is a shared dictionary instance

# agents/agent_workpad.py

from typing import Dict, List, Any, TypedDict

# class State(TypedDict, total=False):
#     pass

# # AgentWorkpad is a shared dictionary instance
# AgentWorkpad: Dict[str, List[Any]] = {}

def create_state_typed_dict(agent_team):
    """
    Creates a TypedDict 'State' where keys are agent names and values are List[str].
    """
    from typing import TypedDict, List

    # Build the fields for the TypedDict dynamically
    fields = {}
    for agent_class in agent_team:
        agent_name = agent_class.name  # Access the 'name' attribute
        fields[agent_name] = List[str]

    # Create the TypedDict 'State' with the dynamic fields
    State = TypedDict('State', fields, total=False)
    return State