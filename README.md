# Meta Expert

A project for versatile AI agents that can run with proprietary models or completely open-source. The meta expert has two agents: a basic [Meta Agent](Docs/Meta-Prompting%20Overview.MD), and [Jar3d](Docs/Introduction%20to%20Jar3d.MD), a more sophisticated and versatile agent.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Prerequisites](#prerequisites)
   - [Environment Setup](#environment-setup)
3. [Repository Setup](#repository-setup)
4. [API Key Configuration](#api-key-configuration)
5. [Basic Meta Agent Setup](#setup-for-basic-meta-agent)
6. [Jar3d Setup](#setup-for-jar3d)
   - [Setting up the NLM-Ingestor Server](#1-setting-up-the-nlm-ingestor-server)
7. [Working with Ollama](#if-you-want-to-work-with-ollama)
8. [On the Roadmap for Jar3d](#on-the-roadmap-for-jar3d)

## Core Concepts

This project leverages three core concepts:

1. Meta prompting: For more information, refer to the paper on **Meta-Prompting** ([source](https://arxiv.black/pdf/2401.12954)). Read our notes on [Meta-Prompting Overview](Docs/Meta-Prompting%20Overview.MD) for a more concise overview.
2. Chain of Reasoning: For [Jar3d](#setup-for-jar3d), we also leverage an adaptation of [Chain-of-Reasoning](https://github.com/ProfSynapse/Synapse_CoR)
3. [Jar3d](#setup-for-jar3d) uses retrieval augmented generation, which isn't used within the [Basic Meta Agent](#setup-for-basic-meta-agent). Read our notes on [Overview of Agentic RAG](Docs/Overview%20of%20Agentic%20RAG.MD).

## Prerequisites

### Environment Setup

1. **Install Anaconda:**  
   Download Anaconda from [https://www.anaconda.com/](https://www.anaconda.com/).

2. **Create a Virtual Environment:**

   ```bash
   conda create -n agent_env python=3.11 pip
   ```

3. **Activate the Virtual Environment:**

   ```bash
   conda activate agent_env
   ```

## Repository Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/brainqub3/meta_expert.git
   ```

2. **Navigate to the Repository:**

   ```bash
   cd /path/to/your-repo/meta_expert
   ```

3. **Install Requirements:**

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Create your `config.yaml` file:**

   ```bash
   cp config/config.example.yaml config/config.yaml
   ```

2. **Open the `config.yaml` file:**

   ```bash
   nano config/config.yaml
   ```

### API Key Configuration

Enter API Keys:

- **Serper API Key:** Get it from [https://serper.dev/](https://serper.dev/)
- **OpenAI API Key:** Get it from [https://openai.com/](https://openai.com/)
- **Gemini API Key:** Get it from [https://ai.google.dev/gemini-api](https://ai.google.dev/gemini-api)
- **Claude API Key:** Get it from [https://docs.anthropic.com/en/api/getting-started](https://docs.anthropic.com/en/api/getting-started)
- **Groq API Key:** Get it from [https://console.groq.com/keys](https://console.groq.com/keys)

### Endpoints Configuration

Set the `LLM_SERVER` variable to choose your inference provider. Possible values are:

- openai
- mistral
- claude
- ollama
- groq
- vllm

Example:

```yaml
LLM_SERVER: ollama
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

*Note: Currently, th best results are with Claude 3.5 Sonnet and Llama 3.1 70B. Results with GPT-4o are inconsistent*

Try Jar3d with:

- Writing a newsletter - [Example](Docs/Example%20Outputs/Llama%203.1%20Newsletter.MD)
- Writing a literature review
- As a holiday assistant

Jar3d is in active development, and its capabilities are expected to improve with better models. Feedback is greatly appreciated.

Jar3d is capable of running 100% locally. However, the setup itself is non-trivial. There are two components to set up:

1. The nlm-ingestor server
2. Running the application

### 1. Setting up the NLM-Ingestor Server

Running Jar3d requires you to set up your own backend server to run the document parsing. We leverage the [nlm-ingestor](https://github.com/nlmatics/nlm-ingestor) and [llm-sherpa](https://github.com/nlmatics/llmsherpa?tab=readme-ov-file) from NLMatics to do this.

The nlm-ingestor uses a modified version of Apache Tika for parsing documents. The server will be deployed locally on whatever machine you run the Docker image.

*The server provides an easy way to parse and intelligently chunk a variety of documents including "HTML", "PDF", "Markdown", and "Text". There is an option to turn on OCR; check out the [docs](https://github.com/nlmatics/nlm-ingestor#:~:text=to%20apply%20OCR%20add%20%26applyOcr%3Dyes).*

#### Setup Steps

1. Ensure you have [Docker](https://www.docker.com/) installed on your machine. Once installed, ensure you have started the Docker daemon.

2. Next, pull the Docker image from nlmatics:

   ```bash
   docker pull jamesmtc/nlm-ingestor:latest
   ```

   *Note this version of the docker image is unofficial, it is being used as a stop-gap until the library authors fix bugs in the official version*

3. Once you have pulled the image, you can run the container:

   ```bash
   docker run -p 5010:5001 jamesmtc/nlm-ingestor:latest
   ```

4. Navigate to `config/config.yaml` and check that the `LLM_SHERPA_SERVER` variable is exactly like this:

   ```
   http://localhost:5010/api/parseDocument?renderFormat=all
   ```

   *Note: You can change the port mapping from 5010 to whatever mapping you want. You must ensure that it is consistent with the mapping you select in the `docker run` command.*

5. Once you're ready, you can kickstart the Jar3d web-app by running the following from the meta_expert directory.

   ```bash
   chainlit run chat.py
   ```

#### Interacting with Jar3d

Once you're setup, Jar3d will proceed to introduce itself and ask some questions. The questions are designed to help you refine your requirements. When you feel you have provided all the relevant information to Jar3d, you can end the questioning part of the workflow play typing `/end`.

## If you want to work with Ollama

Jar3d now supports easier integration with Ollama, including configuration via `config.yaml` and automatic model pulling.

### Setup Ollama Server

1. **Download Ollama:**
   Download from [https://ollama.com/download](https://ollama.com/download)

2. **Start the Ollama server:**
   Follow the instructions for your operating system to start the Ollama server.

### Configure Jar3d for Ollama

1. **Edit `config/config.yaml`:**
   Add the following lines to configure Ollama:

   ```yaml
   OLLAMA_HOST: http://localhost:11434
   OLLAMA_MODEL: llama3.1:8b
   ```

   If you're using Docker, use `http://host.docker.internal:11434` for `OLLAMA_HOST`.

2. **Automatic Model Pulling:**
   Jar3d will now automatically pull the specified Ollama model if it's not already available. This happens when you first try to use the model.

### Manual Model Management (Optional)

You can still manually manage Ollama models if needed:

1. **Pull an Ollama Model:**

   ```bash
   curl http://localhost:11434/api/pull -d "{\"name\": \"llama3\"}"
   ```

2. **List Available Models:**

   ```bash
   curl http://localhost:11434/api/tags
   ```

For more information on Ollama's API and available models, refer to the [Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models).

### Notes

- Ensure Ollama is running before starting Jar3d.
- The first use of a new model might take some time as Jar3d pulls it from Ollama.
- You can change the model anytime by updating `OLLAMA_MODEL` in `config.yaml`.
For more information, refer to the Ollama [API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models).

## Docker Setup for Jar3d

Jar3d can be run using Docker for easier setup and deployment.

### Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Quick Start

1. **Clone the repository:**

   ```bash
   git clone https://github.com/brainqub3/meta_expert.git
   cd meta_expert
   ```

2. **Build and start the containers:**

   ```bash
   docker-compose up --build
   ```

3. **Access Jar3d:**
   Once running, access the Jar3d web interface at `http://localhost:8000`.

### Notes

- The Docker setup includes Jar3d and the NLM-Ingestor service.
- Playwright and its browser dependencies are included for web scraping capabilities.
- Ollama is not included in this Docker setup. If needed, set it up separately and configure in `config.yaml`.
- Configuration is handled through `config.yaml`, not environment variables in docker-compose.

For troubleshooting, check the container logs:

```bash
docker-compose logs
```

Refer to the project's GitHub issues for common problems and solutions.

## On the Roadmap for Jar3d

- Feedback to Jar3d so that final responses can be iterated on and amended.
- Long term memory.
- Frontend.
- Integrations to RAG platforms for more intelligent document processing and faster RAG.
