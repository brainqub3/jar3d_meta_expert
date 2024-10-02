from termcolor import colored
from agents.agent_workpad import create_state_typed_dict
from langgraph.graph import StateGraph, END, START, MessagesState 
from langgraph.checkpoint.memory import MemorySaver
import json

def build_workflow(agent_team, requirements):
    # Ensure 'meta_agent' and 'reporter_agent' are in agent_team
    agent_names = [agent.name for agent in agent_team]
    if 'meta_agent' not in agent_names or 'reporter_agent' not in agent_names:
        raise ValueError("Both 'meta_agent' and 'reporter_agent' must be in agent_team")

    # Create the State subclass
    State = create_state_typed_dict(agent_team)

    # Initialize the state
    state = State()

    # Register the agents with the state
    for agent in agent_team:
        agent.register(state)

    print(colored(f"\n\nDEBUG: State: {State}\n\n", 'red'))
    print(colored(f"\nInitial state:\n\n{state}\n\n", 'blue'))

    # Define the graph
    graph = StateGraph(State)

    # Dictionary to map agent names to node names
    agent_nodes = {}

    # Add nodes dynamically for each agent
    for agent in agent_team:
        node_name = f"{agent.name}_node"
        agent_nodes[agent.name] = node_name
        if agent.name == 'meta_agent':
            # For meta_agent, pass requirements
            graph.add_node(node_name, lambda state, agent=agent: agent.invoke(state=state, requirements=requirements))
        else:
            graph.add_node(node_name, lambda state, agent=agent: agent.invoke(state=state))

    # Define the routing function
    def routing_function(state):
        # print(colored(f"\n\nDEBUG: State: {state}\n\n", 'red'))
        if state.get("meta_agent", ""):
            meta_agent_response = state.get("meta_agent", "")[-1].page_content
            try:
                meta_agent_response_json = json.loads(meta_agent_response)
                next_agent = meta_agent_response_json.get("Agent")
                next_agent_node = agent_nodes.get(next_agent, END)
            except json.JSONDecodeError:
                next_agent_node = END
        else:
            next_agent_node = END
        print(colored(f"\n\nDEBUG: Next agent: {next_agent_node}\n\n", 'red'))
        return next_agent_node

    # Edge from START to meta_agent_node
    graph.add_edge(START, agent_nodes['meta_agent'])

    # Conditional edge from meta_agent_node to next agent
    graph.add_conditional_edges(
        agent_nodes['meta_agent'],
        lambda state: routing_function(state),
    )

    # For each agent, add an edge back to 'meta_agent_node' after the agent's node is processed
    for agent in agent_team:
        node_name = agent_nodes[agent.name]
        if agent.name != 'reporter_agent' and agent.name != 'meta_agent':
            graph.add_edge(node_name, agent_nodes['meta_agent'])
        elif agent.name == 'reporter_agent':
            # 'reporter_agent_node' goes to END
            graph.add_edge(node_name, END)

    # Compile the workflow
    checkpointer = MemorySaver()
    workflow = graph.compile(checkpointer)
    return workflow, state