from langchain_core.messages import AIMessage

def get_ai_message_contents(conversation_history):
    return [message.content for message in conversation_history if isinstance(message, AIMessage)]