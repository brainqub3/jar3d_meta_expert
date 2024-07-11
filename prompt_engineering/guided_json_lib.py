guided_json_search_query = {
  "type": "object",
  "properties": {
    "search_query": {
      "type": "string",
      "description": "The refined Google search engine query that aligns with the response from your managers."
    }
  },
  "required": ["search_query"]
}

guided_json_best_url = {
  "type": "object",
  "properties": {
    "best_url": {
      "type": "string",
      "description": "The URL of the Serper results that aligns most with the instructions from your manager."
    }
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
