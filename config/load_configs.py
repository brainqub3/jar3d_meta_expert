import os
import yaml

def load_config(file_path):
    default_values = {
        'SERPER_API_KEY': 'default_serper_api_key',
        'OPENAI_API_KEY': 'default_openai_api_key',
        'CLAUDE_API_KEY': 'default_claude_api_key',
        'GEMINI_API_KEY': 'default_gemini_api_key',
        'GROQ_API_KEY': 'default_groq_api_key',
    }
    
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        for key, value in config.items():
            if not value:
                os.environ[key] = default_values.get(key, '')
            else:
                os.environ[key] = value