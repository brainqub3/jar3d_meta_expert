# Meta Expert

A project for versatile AI agents that can run with proprietary models or completely open-source. The meta expert has two agents: a basic [Meta Agent](Docs/Meta-Prompting%20Overview.MD), and [Jar3d](Docs/Introduction%20to%20Jar3d.MD), a more sophisticated and versatile agent.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Prerequisites](#prerequisites)
3. [Configuration](#configuration)
   - [API Key Configuration](#api-key-configuration)
   - [Endpoints Configuration](#endpoints-configuration)
4. [Setup for Basic Meta Agent](#setup-for-basic-meta-agent)
5. [Setup for Jar3d](#setup-for-jar3d)
   - [Docker Setup for Jar3d](#docker-setup-for-jar3d)
   - [Interacting with Jar3d](#interacting-with-jar3d)
6. [Roadmap for Jar3d](#roadmap-for-jar3d)

## Core Concepts

This project leverages four core concepts:

1. Meta prompting: For more information, refer to the paper on **Meta-Prompting** ([source](https://arxiv.black/pdf/2401.12954)). Read our notes on [Meta-Prompting Overview](Docs/Meta-Prompting%20Overview.MD) for a more concise overview.
2. Chain of Reasoning: For [Jar3d](#setup-for-jar3d), we also leverage an adaptation of [Chain-of-Reasoning](https://github.com/ProfSynapse/Synapse_CoR)
3. [Jar3d](#setup-for-jar3d) uses retrieval augmented generation, which isn't used within the [Basic Meta Agent](#setup-for-basic-meta-agent). Read our notes on [Overview of Agentic RAG](Docs/Overview%20of%20Agentic%20RAG.MD). 
4. Jar3d can generate knowledge graphs from web-pages allowing it to produce more comprehensive outputs.

## Prerequisites

1. Clone this project to your work environment/local directory:
   ```bash
   git clone https://github.com/brainqub3/meta_expert.git
   ```

2. You will need Docker and Docker Composed installed to get the project up and running:
   - [Docker](https://www.docker.com/get-started)
   - [Docker Compose](https://docs.docker.com/compose/install/)

3. **If you wish to use Hybrid Retrieval, you will need to create a Free Neo4j Aura Account:**
   - [Neo4j Aura](https://neo4j.com/)

## Configuration

1. Navigate to the Repository:
   ```bash
   cd /path/to/your-repo/meta_expert
   ```

2. Open the `config.yaml` file:
   ```bash
   nano config/config.yaml
   ```

### API Key Configuration

Enter API Keys for your choice of LLM provider:

- **Serper API Key:** Get it from [https://serper.dev/](https://serper.dev/)
- **OpenAI API Key:** Get it from [https://openai.com/](https://openai.com/)
- **Gemini API Key:** Get it from [https://ai.google.dev/gemini-api](https://ai.google.dev/gemini-api)
- **Claude API Key:** Get it from [https://docs.anthropic.com/en/api/getting-started](https://docs.anthropic.com/en/api/getting-started)
- **Groq API Key:** Get it from [https://console.groq.com/keys](https://console.groq.com/keys)

*For Hybrid retrieval, you will require a Claude API key*

### Endpoints Configuration

Set the `LLM_SERVER` variable to choose your inference provider. Possible values are:

- openai
- mistral
- claude
- gemini (Not currently supported)
- ollama (Not currently supported)
- groq
- vllm (Not currently supported)

Example:

```yaml
LLM_SERVER: claude
```

Remember to keep your `config.yaml` file private as it contains sensitive information.

## Setup for Basic Meta Agent

The basic meta agent is an early iteration of the project. It demonstrates meta prompting rather than being a useful tool for research. It uses a naive approach of scraping the entirety of a web page and feeding that into the context of the meta agent, who either continues the task or delivers a final answer.

### Run Your Query in Shell

```bash
python -m agents.meta_agent
```

Then enter your query.

## Setup for Jar3d

Jar3d is a more sophisticated agent that uses RAG, Chain-of-Reasoning, and Meta-Prompting to complete long-running research tasks.

*Note: Currently, the best results are with Claude 3.5 Sonnet and Llama 3.1 70B. Results with GPT-4 are inconsistent*

Try Jar3d with:

- Writing a newsletter - [Example](Docs/Example%20Outputs/Llama%203.1%20Newsletter.MD)
- Writing a literature review
- As a holiday assistant

Jar3d is in active development, and its capabilities are expected to improve with better models. Feedback is greatly appreciated.

### Docker Setup for Jar3d

Jar3d can be run using Docker for easier setup and deployment.

#### Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

#### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/brainqub3/meta_expert.git
   cd meta_expert
   ```

2. Build and start the containers:
   ```bash
   docker-compose up --build
   ```

3. Access Jar3d:
   Once running, access the Jar3d web interface at `http://localhost:8000`.

You can end your docker session by pressing `Ctrl + C` or `Cmd + C` in your terminal and running:
```bash
docker-compose down
```

#### Notes

- The Docker setup includes Jar3d and the NLM-Ingestor service.
- Playwright and its browser dependencies are included for web scraping capabilities.
- Ollama is not included in this Docker setup. If needed, set it up separately and configure in `config.yaml`.
- Configuration is handled through `config.yaml`, not environment variables in docker-compose.

For troubleshooting, check the container logs:

```bash
docker-compose logs
```

Refer to the project's GitHub issues for common problems and solutions.

### Interacting with Jar3d

Once you're set up, Jar3d will proceed to introduce itself and ask some questions. The questions are designed to help you refine your requirements. When you feel you have provided all the relevant information to Jar3d, you can end the questioning part of the workflow by typing `/end`.

## Roadmap for Jar3d

- Feedback to Jar3d so that final responses can be iterated on and amended.
- Long-term memory.
- Full Ollama and vLLM integration.
- Integrations to RAG platforms for more intelligent document processing and faster RAG.