# Introduction to Jar3d

## Problem Statement
The goal was to develop an AI agent capable of leveraging the full potential of both proprietary and open-source models for research-intensive tasks.

## What is Jar3d?
Jar3d is a versatile agent that combines [chain-of-reasoning](https://github.com/ProfSynapse/Synapse_CoR), Meta-Prompting, and Agentic RAG techniques.

- It features integrations with popular providers and open-source models, allowing for 100% local operation given sufficient hardware resources.

### Use Cases & Applications
- Long-running research tasks, writing literature reviews, newsletters, etc.
- Potential adaptation for use with internal company documents, requiring no internet access.
- Can function as a research assistant or a local version of services like Perplexity.

For setup instructions, please refer to the [Setup Jar3d] guide.

## Prompt Engineering
Jar3d utilizes two powerful prompts written entirely in Markdown:
1. [Jar3d Meta-Prompt]
2. [Jar3d Requirements Prompt]

Both prompts incorporate adaptations of the Chain of Reasoning technique.

## Jar3d Architecture
The Jar3d architecture incorporates aspects of Meta-Prompting, Agentic RAG, and an adaptation of [Chain of Reasoning](https://github.com/ProfSynapse/Synapse_CoR).

```mermaid
graph TD
    A[Jar3d] -->|Gathers requirements| B[MetaExpert]
    B -->|Uses chain of reasoning| C{Router}
    C -->|Tool needed| D[Tool Expert]
    C -->|No tool needed| E[Non-Tool Expert]
    D -->|Internet research & RAG| F[Result]
    E -->|Writer or Planner| G[Result]
    F --> B
    G --> B
    B --> C
    C -->|Final output| I[Deliver to User]
    C -->|Needs more detail| B
    
    subgraph "Jar3d Process"
    J[Start] --> K[Gather Requirements]
    K --> L{Requirements Adequate?}
    L -->|No| K
    L -->|Yes| M[End Jar3d Process]
    end
    
    A -.-> J
    M -.-> B
```