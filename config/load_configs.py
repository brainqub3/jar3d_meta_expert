import os
import yaml
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        for key, value in config.items():
            os.environ[key] = str(value)
            # logger.debug(f"Set environment variable: {key}={value} (type: {type(value)})")
        
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")