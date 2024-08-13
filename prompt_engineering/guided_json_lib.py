guided_json_search_query = {
  "type": "object",
  "properties": {
    "search_queries": {
      "type": "array",
      "items":{"type": "string"},
      "description": "List of generated search queries"
    }
  },
  "required": ["search_query"]
}


guided_json_search_query_two = {
  "type": "object",
  "properties": {
    "search_queries": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "engine": {
            "type": "string",
            "enum": ["search", "shopping"],
            "description": "The search engine to use (either 'search' or 'shopping')"
          },
          "query": {
            "type": "string",
            "description": "The search query string"
          }
        },
        "required": ["engine", "query"]
      },
      "minItems": 1,
      "description": "List of generated search queries with their corresponding engines"
    }
  },
  "required": ["search_queries"]
}

guided_json_best_url = {
  "type": "object",
  "properties": {
    "best_url": {
      "type": "string",
      "description": "The URL of the Serper results that aligns most with the instructions from your manager."
    },
    "pdf": {
      "type": "boolean",
      "description": "A boolean value indicating whether the URL is a PDF or not. This should be True if the URL is a PDF, and False otherwise."
    }
  },
  "required": ["best_url", "pdf"]
}


guided_json_best_url_two = {
  "type": "object",
  "properties": {
    "best_url": {
      "type": "string",
      "description": "The URL of the Serper results that aligns most with the instructions from your manager."
    },
  },
  "required": ["best_url"]
}


guided_json_router_decision = {
  "type": "object",
  "properties": {
    "router_decision": {
      "type": "string",
      "description": "Return the next agent to pass control to."
    }
  },
  "required": ["router_decision"]
}


guided_json_parse_expert = {
  "type": "object",
  "properties": {
    "expert": {
      "type": "string",
      "description": "Expert Planner or Expert Writer"
    }
  },
  "required": ["expert"]
}
